import os
import sys
import time
import tqdm
import cv2
import math
import torch
import numpy as np
import pandas as pd
import tensorrt as trt
import steps_analysis
from torchvision import transforms
from scipy.ndimage import gaussian_filter
from mmpose.apis.inference import dataset_meta_from_config
from mmpose.utils import adapt_mmdet_pipeline
from mmdet.apis import inference_detector, init_detector
from typing import Union, Optional, Sequence, Dict, Any, List, Tuple


class TRTWrapper(torch.nn.Module):
    def __init__(self, engine: Union[str, trt.ICudaEngine],
                 output_names: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        self.engine = engine
        if isinstance(self.engine, str):
            with trt.Logger() as logger, trt.Runtime(logger) as runtime:
                with open(self.engine, mode='rb') as f:
                    engine_bytes = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()
        names = [_ for _ in self.engine]
        input_names = list(filter(self.engine.binding_is_input, names))
        self._input_names = input_names
        self._output_names = output_names

        if self._output_names is None:
            output_names = list(set(names) - set(input_names))
            self._output_names = output_names

    def forward(self, inputs: Dict[str, torch.Tensor]):
        assert self._input_names is not None
        assert self._output_names is not None
        bindings = [None] * (len(self._input_names) + len(self._output_names))
        profile_id = 0
        for input_name, input_tensor in inputs.items():
            # check if input shape is valid
            profile = self.engine.get_profile_shape(profile_id, input_name)
            assert input_tensor.dim() == len(
                profile[0]), 'Input dim is different from engine profile.'
            for s_min, s_input, s_max in zip(profile[0], input_tensor.shape,
                                             profile[2]):
                assert s_min <= s_input <= s_max, \
                    'Input shape should be between ' \
                    + f'{profile[0]} and {profile[2]}' \
                    + f' but get {tuple(input_tensor.shape)}.'
            idx = self.engine.get_binding_index(input_name)

            # All input tensors must be gpu variables
            assert 'cuda' in input_tensor.device.type
            input_tensor = input_tensor.contiguous()
            if input_tensor.dtype == torch.long:
                input_tensor = input_tensor.int()
            self.context.set_binding_shape(idx, tuple(input_tensor.shape))
            bindings[idx] = input_tensor.contiguous().data_ptr()

        # create output tensors
        outputs = {}
        for output_name in self._output_names:
            idx = self.engine.get_binding_index(output_name)
            dtype = torch.float32
            shape = tuple(self.context.get_binding_shape(idx))
            device = torch.device('cuda')
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[output_name] = output
            bindings[idx] = output.data_ptr()
        self.context.execute_async_v2(bindings,
                                      torch.cuda.current_stream().cuda_stream)
        return outputs


def dispose_bbox(bboxes, img):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
    ])

    # bboxes = bboxes.cpu().numpy()

    # ratio = img_info["ratio"]
    # img = img_info["raw_img"]

    # bboxes = filter_bboxes(bboxes)

    if bboxes is None or len(bboxes) == 0:
        return None, None

    # bboxes = bboxes.astype(np.int32)
    # bboxes[:, 0] -= 5
    # bboxes[:, 1] += 5
    # bboxes[:, 2] -= 5
    # bboxes[:, 3] += 2

    new_boxes = torch.empty((bboxes.shape[0], 4), dtype=torch.int32, device='cuda')
    new_images = torch.empty((bboxes.shape[0], 3, 256, 192))

    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        x1 = (x1 - 5).astype(np.int32)
        x2 = (x2 + 5).astype(np.int32)
        y1 = (y1 - 5).astype(np.int32)
        y2 = (y2 + 2).astype(np.int32)

        if x2 > img.shape[1]:
            x2 = img.shape[1]
        if y2 > img.shape[0]:
            y2 = img.shape[0]
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0

        correction_factor = 256 / 192 * (x2 - x1) / (y2 - y1)
        if correction_factor >= 1:
            center = y1 + (y2 - y1) // 2
            length = int(round(y2 - y1) * correction_factor)  # 满足比例的新的高度
            y1_new = int(center - length // 2)
            y2_new = int(center + length // 2)
            image_crop = img[y1:y2, x1:x2, ::-1]
            image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            pad = (int(abs(y1_new - y1)), int(abs(y2_new - y2)))
            image_crop = np.pad(image_crop, (pad, (0, 0), (0, 0)))
            new_images[i] = transform(image_crop)
            new_boxes[i] = torch.tensor([x1, y1_new, x2, y2_new])

        elif correction_factor < 1:
            center = x1 + (x2 - x1) // 2
            length = int(round(x2 - x1) * 1 / correction_factor)
            x1_new = int(center - length // 2)
            x2_new = int(center + length // 2)
            image_crop = img[y1:y2, x1:x2, ::-1]
            image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            pad = (int(abs(x1_new - x1)), int(abs(x2_new - x2)))
            image_crop = np.pad(image_crop, ((0, 0), pad, (0, 0)))
            new_images[i] = transform(image_crop)
            new_boxes[i] = torch.tensor([x1_new, y1, x2_new, y2])

    return new_images, new_boxes


def draw_polygons(img, polygons, edge_colors, face_colors, alpha=0.6):
    new_img = img.copy()
    new_img = cv2.fillConvexPoly(new_img, polygons, edge_colors)
    img = cv2.addWeighted(img, 1 - alpha, new_img, alpha, 0)

    return img


def transform_keypoints(keypoints, scores, bboxes, original_img):
    """
    将推理结果中的关键点坐标从裁剪图像空间转换到原图空间，并将关键点与分数组合为 (N, 23, 3) 数组。

    keypoints: tensor, 推理结果中的关键点坐标，形状为 (M, 23, 2)，表示M个目标，每个目标23个关键点的 (x, y) 坐标。
    scores: tensor, 每个关键点的置信度分数，形状为 (M, 23)，表示M个目标，每个目标23个关键点的置信度。
    bboxes: tensor, 包含M个bbox的坐标，形状为 (M, 4)，格式为 [x1, y1, x2, y2]。

    返回:
        combined_keypoints: tensor, 形状为 (M, 23, 3)，包含每个目标的关键点坐标和置信度，最后一维为 [x, y, score]。
    """
    M, num_keypoints, _ = keypoints.shape  # M为目标数，num_keypoints为关键点数，通常为23
    img_height, img_width = original_img.shape[:2]

    # 初始化一个 (M, 23, 3) 的 tensor，存储每个目标的 [x, y, score]
    combined_keypoints = torch.zeros((M, num_keypoints, 3), device='cuda')

    for m in range(M):
        x1, y1, x2, y2 = bboxes[m].tolist()

        # 将关键点从裁剪图像坐标系变换为原图坐标系
        scale_x = (x2 - x1) / 192
        scale_y = (y2 - y1) / 256

        for i in range(num_keypoints):
            # 获取当前关键点的 x 和 y 坐标
            x, y = keypoints[m, i, :]

            # 将关键点坐标变换为原图坐标
            orig_x = int(x * scale_x + x1)
            orig_y = int(y * scale_y + y1)

            # 获取当前关键点的置信度分数
            score = scores[m, i].item()

            # 将 [x, y, score] 存储到数组
            combined_keypoints[m, i, 0] = orig_x
            combined_keypoints[m, i, 1] = orig_y
            combined_keypoints[m, i, 2] = score

    return combined_keypoints


def visualizer(img, pose_result, radius=8, thickness=5, kpt_score_thr=0.3):
    """
    nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder,
    left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip,
    left_knee, right_knee, left_ankle, right_ankle, lb, ls, lh, rb, rs, rh, neck
    """

    keypoints_color_info = np.array([
        [255, 0, 0], [255, 0, 255], [170, 0, 255], [255, 0, 85], [255, 0, 170],
        [85, 255, 0], [255, 170, 0], [0, 255, 0], [255, 255, 0], [0, 255, 85],
        [170, 255, 0], [0, 85, 255], [0, 255, 170], [0, 0, 255], [0, 255, 255],
        [85, 0, 255], [0, 170, 255], [130, 0, 255], [155, 0, 255], [225, 0, 255],
        [0, 85, 255], [0, 65, 255], [0, 30, 255], [255, 85, 0]
    ])
    skeleton = [
        [0, 1], [0, 2], [1, 3], [2, 4], [23, 0], [23, 5], [23, 6], [23, 11],
        [23, 12], [5, 7], [6, 8], [7, 9], [8, 10], [11, 13], [12, 14], [13, 15],
        [14, 16], [15, 17], [15, 18], [15, 19], [16, 20], [16, 21], [16, 22]
    ]
    skeleton_color_info = np.array([
        [255, 0, 255], [255, 0, 170], [255, 0, 170], [170, 0, 255], [0, 0, 255],
        [255, 85, 0], [255, 0, 0], [0, 255, 255], [0, 255, 0], [170, 255, 0],
        [255, 270, 0], [85, 255, 0], [255, 255, 0], [0, 170, 255], [0, 255, 85],
        [0, 85, 255], [0, 255, 170], [0, 45, 255], [0, 32, 255], [0, 16, 255],
        [0, 255, 255], [32, 255, 255], [64, 255, 255]
    ])

    kpts_info = pose_result[0]
    # print(f"kpts_info: {kpts_info.shape}")
    keypoints = kpts_info[:, :, :2]
    # print(f"keypoints: {keypoints.shape}")
    scores = kpts_info[:, :, 2]
    # print(f"score: {scores.shape}")
    kpts_visible = np.ones(keypoints.shape[1]).reshape(1, -1)
    # print(f"kpts_visible: {kpts_visible.shape}")
    keypoints_info = np.concatenate((keypoints, scores[..., None], kpts_visible[..., None]), axis=-1)

    neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
    neck[:, 2:4] = np.logical_and(keypoints_info[:, 5, 2:4] > kpt_score_thr,
                                  keypoints_info[:, 6, 2:4] > kpt_score_thr).astype(int)

    new_keypoints_info = np.insert(keypoints_info, 23, neck, axis=1)

    keypoints, scores, keypoints_visible = new_keypoints_info[..., :2], new_keypoints_info[..., 2], new_keypoints_info[
        ..., 3]

    for kpts, score, visible in zip(keypoints, scores, keypoints_visible):
        kpts = np.array(kpts, copy=False)

        for sk_id, sk in enumerate(skeleton):
            pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
            pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))
            if not (visible[sk[0]] and visible[sk[1]]):
                continue

            if (pos1[0] <= 0 or pos1[0] >= img.shape[1] or pos1[1] <= 0 or pos1[1] >= img.shape[0] or
                    pos2[0] <= 0 or pos2[0] >= img.shape[1] or pos2[1] <= 0 or pos2[1] >= img.shape[0] or
                    score[sk[0]] <= kpt_score_thr or score[sk[1]] <= kpt_score_thr):
                continue

            X = np.array((pos1[0], pos2[0]))
            Y = np.array((pos1[1], pos2[1]))
            color = tuple(skeleton_color_info[sk_id].astype(int).tolist())
            # transparency = 1.0 * max(0, min(1, 0.5 * (score[sk[0]] + score[sk[1]])))

            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            transparency = 0.6
            angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
            polygons = cv2.ellipse2Poly(
                (int(mX), int(mY)),
                (int(length / 2), thickness),
                int(angle), 0, 360, 1
            )

            new_img = img.copy()
            new_img = cv2.fillConvexPoly(new_img, polygons, color)
            img = cv2.addWeighted(img, 1 - transparency, new_img, transparency, 0)

        # draw keypoints
        for kid, kpt in enumerate(kpts):
            if score[kid] < kpt_score_thr or not visible[kid]:
                continue

            color = tuple(keypoints_color_info[kid].astype(int).tolist())
            transparency = 1.0 * max(0, min(1, score[kid]))
            new_img = cv2.circle(img.copy(), (int(kpt[0]), int(kpt[1])), int(radius), color, -1)
            img = cv2.addWeighted(img, 1 - transparency, new_img, transparency, 0)

    return img


def nms(dets: np.ndarray, thr: float):
    """
    Greedily select boxes with high confidence and overlap <= thr.

    Args:
        dets (np.ndarray): [[x1, y1, x2, y2, score]].
        thr (float): Retain overlap < thr.

    Returns:
        list: Indexes to keep.
    """
    if len(dets) == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thr)[0]
        order = order[inds + 1]

    return keep


def resize_and_pad_image(img, target_height=800, target_width=1333):
    """
    Resize input image while keeping the aspect ratio. The height will be set to target_height,
    and the width will be adjusted accordingly. Padding will be applied to make the final image
    size equal to target_width.

    Parameters:
        img (np.array): Input image (H, W, C).
        target_height (int): Target height (default 800).
        target_width (int): Target width (default 1333).

    Returns:
        np.array: Resized and padded image.
        float: Scaling factor for width.
        float: Scaling factor for height.
        tuple: Padding information (top, bottom, left, right).
    """
    h, w, _ = img.shape
    scale_factor = target_height / h
    new_w = int(w * scale_factor)

    # Resize the image keeping aspect ratio
    resized_img = cv2.resize(img, (new_w, target_height))

    # Calculate padding for width
    pad_left = (target_width - new_w) // 2
    pad_right = target_width - new_w - pad_left

    # Pad the image
    padded_img = cv2.copyMakeBorder(resized_img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return padded_img, scale_factor, pad_left, pad_right


def restore_bboxes(bboxes, scale_factor, padding, original_size):
    """
    Restore the bounding boxes from the resized and padded image back to the original image size.

    Parameters:
        bboxes (np.array): Bounding boxes from the model output, shape (N, 4).
        scale_factor (float): The scaling factor for the width.
        padding (tuple): Padding applied to the image (pad_left, pad_right).
        original_size (tuple): Original image size (height, width).

    Returns:
        np.array: Restored bounding boxes with shape (N, 4).
    """
    h, w = original_size
    pad_left, pad_right = padding

    restored_bboxes = []

    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox

        # Undo padding
        xmin = (xmin - pad_left) / scale_factor
        xmax = (xmax - pad_left) / scale_factor

        # Undo resizing
        xmin = max(0, min(w, xmin))
        ymin = max(0, min(h, ymin))
        xmax = max(0, min(w, xmax))
        ymax = max(0, min(h, ymax))

        restored_bboxes.append([xmin, ymin, xmax, ymax])

    return np.array(restored_bboxes)


def detect_inference(detector, img):
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                   pred_instance.scores > 0.7)]
    bboxes = bboxes[nms(bboxes, 0.4), :4]

    return bboxes

def overwrite_video(video, output_root, csv, diff=3):
    k_l = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
           "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
           "left_knee", "right_knee", "left_ankle", "right_ankle", "left_bigtoe",
           "left_smalltoe", "left_heel", "right_bigtoe", "right_smalltoe", "right_heel"]

    keypoints = {}
    pose_result = []
    frame_list = []
    # open the csv info
    kpts_info = pd.read_csv(csv)
    for col in kpts_info.columns:
        if col.startswith("frame"):
            frame_list = kpts_info.loc[:, col].tolist()
        if col.endswith("score"):
            keypoints[col] = kpts_info.loc[:, col].tolist()
        #     continue
        keypoints[col] = gaussian_filter(kpts_info.loc[:, col].tolist(), sigma=1.6)

    for kpt_idx in range(len(keypoints['nose_x'])):
        pose = []
        for label in k_l:
            if label == "neck":
                pose.append([
                    (keypoints["left_shoulder_x"][kpt_idx] + keypoints["right_shoulder_x"][kpt_idx]) / 2,
                    (keypoints["left_shoulder_y"][kpt_idx] + keypoints["right_shoulder_y"][kpt_idx]) / 2,
                    (keypoints["left_shoulder_score"][kpt_idx] + keypoints["right_shoulder_score"][kpt_idx]) / 2,
                ])
                continue
            x = keypoints[label + "_x"][kpt_idx]
            y = keypoints[label + "_y"][kpt_idx]
            s = keypoints[label + "_score"][kpt_idx]
            pose.append([x, y, s])
        pose_result.append(pose)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    progress_bar = tqdm.tqdm(total=frame_count)
    video_writer = cv2.VideoWriter(
        f"{os.path.join(output_root, os.path.splitext(os.path.basename(video))[0])}.mp4",
        cv2.VideoWriter_fourcc(*'mp4v'),
        int(fps / diff), (int(frame_width), int(frame_height))
    )

    frame_idx = 0
    cur_frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        else:
            progress_bar.update(1)
            if frame_idx == int(frame_list[cur_frame_idx]):
                pos_info = np.array([[pose_result[cur_frame_idx]]])
                img = visualizer(frame, pos_info, radius=14, thickness=12)
                video_writer.write(img)
                cur_frame_idx += 1
            if cur_frame_idx >= len(frame_list):
                break

        frame_idx += 1

    video_writer.release()
    cap.release()


def inference(video_path, output_root, difference_bet_frame=3, smooth_sigma=20, num_circle=3, vis=True):
    pose_engine = 'models/rtmpose-m.engine'
    det_config = "config/rtmdet_m_640-8xb32_coco-person.py"
    det_checkpoint = "models/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
    detector = init_detector(det_config, det_checkpoint, device="cuda:0")
    # detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    model = TRTWrapper(pose_engine)

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    output_csv = open(f"{os.path.join(output_root, os.path.splitext(os.path.basename(video_path))[0])}.csv", "w")
    output_csv.write(
        "frame_id,nose_x,nose_y,nose_score,left_eye_x,left_eye_y,left_eye_score,"
        "right_eye_x,right_eye_y,right_eye_score,left_ear_x,left_ear_y,left_ear_score,"
        "right_ear_x,right_ear_y,right_ear_score,left_shoulder_x,left_shoulder_y,left_shoulder_score,"
        "right_shoulder_x,right_shoulder_y,right_shoulder_score,left_elbow_x,left_elbow_y,left_elbow_score,"
        "right_elbow_x,right_elbow_y,right_elbow_score,left_wrist_x,left_wrist_y,left_wrist_score,"
        "right_wrist_x,right_wrist_y,right_wrist_score,left_hip_x,left_hip_y,left_hip_score,"
        "right_hip_x,right_hip_y,right_hip_score,left_knee_x,left_knee_y,left_knee_score,"
        "right_knee_x,right_knee_y,right_knee_score,left_ankle_x,left_ankle_y,left_ankle_score,"
        "right_ankle_x,right_ankle_y,right_ankle_score,left_bigtoe_x,left_bigtoe_y,left_bigtoe_score,"
        "left_smalltoe_x,left_smalltoe_y,left_smalltoe_score,left_heel_x,left_heel_y,left_heel_score,"
        "right_bigtoe_x,right_bigtoe_y,right_bigtoe_score,right_smalltoe_x,right_smalltoe_y,right_smalltoe_score,"
        "right_heel_x,right_heel_y,right_heel_score\n"
    )

    cap = cv2.VideoCapture(video_path)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    progress_bar = tqdm.tqdm(total=frame_count)

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if ret:
            progress_bar.update(1)
            if frame_idx % difference_bet_frame == 0:
                bboxes = detect_inference(detector, frame)
                new_images, new_bboxes = dispose_bbox(bboxes, frame)
                # new_images -> B, 3, 256, 192
                if new_images is not None:
                    output = model(dict(input=new_images[:1, ...].cuda()))
                    kpts = transform_keypoints(output['722'], output['714'], new_bboxes, frame).cpu().numpy()
                    # new_img = visualizer(frame, [kpts], radius=14, thickness=12)

                    # output_video.write(new_img)

                    csv_info = f"{frame_idx},"
                    for idx, kpt in enumerate(kpts[0]):
                        csv_info += str(kpt[0]) + "," + str(kpt[1]) + "," + str(kpt[2]) + ","
                    csv_info = csv_info[:-1] + "\n"
                    output_csv.write(csv_info)
        else:
            break

        frame_idx += 1

    output_csv.close()
    progress_bar.close()
    cap.release()

    if vis:
        overwrite_video(video_path,output_root,
                    f"{os.path.join(output_root, os.path.splitext(os.path.basename(video_path))[0])}.csv",
                    diff=difference_bet_frame)
    try:
        analysis_result = steps_analysis.main_analysis(f"{os.path.join(output_root, os.path.splitext(os.path.basename(video_path))[0])}.csv",
                                 smooth_sigma=smooth_sigma, num_circle=num_circle)
    except Exception as e:
        print(e)
        return None

    return analysis_result


# if __name__ == '__main__':
#     # video_path = sys.argv[1]
#     # output_root = sys.argv[2]
#     # difference_bet_frame = int(sys.argv[3])
#     video_path = "/home/zhaosilei/Projects/PycharmProjects/mmlab/mmpose-1.3.1/tools/workdirs/rtmpose-m_coco25/steps_analysis/73_raw.MP4"
#     output_root = "/home/zhaosilei/Projects/PycharmProjects/mmlab/mmpose-1.3.1/tools/workdirs/rtmpose-m_coco25/output"
#     inference(video_path, output_root, difference_bet_frame=1)
