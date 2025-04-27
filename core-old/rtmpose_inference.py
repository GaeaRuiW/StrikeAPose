import os
import sys
import time
import tqdm
import cv2
import math
import torch
import warnings
import numpy as np
import pandas as pd
# import tensorrt as trt
import steps_analysis
# from torchvision import transforms
from scipy.ndimage import gaussian_filter
from mmpose.apis import inference_topdown
from mmpose.utils import adapt_mmdet_pipeline
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import init_model as init_pose_estimator
from mmpose.apis.inference import dataset_meta_from_config
from mmpose.structures import merge_data_samples, split_instances
from typing import Union, Optional, Sequence, Dict, Any, List, Tuple

warnings.filterwarnings("ignore")


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


def overwrite_video(video, output_root, csv, diff=3):

    k_l = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
           "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
           "left_knee", "right_knee", "left_ankle", "right_ankle", "left_bigtoe",
           "left_smalltoe", "left_heel", "right_bigtoe", "right_smalltoe", "right_heel"]

    keypoints = {}
    pose_result = []
    frame_list = []
    # open the csv info
    kpts_info = pd.read_csv(csv, index_col=False)
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


def process_one_image(img,
                      detector,
                      pose_estimator):

    # predict bbox
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                   pred_instance.scores > 0.7)]
    bboxes = bboxes[nms(bboxes, 0.4), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # # show the results
    # if isinstance(img, str):
    #     img = mmcv.imread(img, channel_order='rgb')
    # elif isinstance(img, np.ndarray):
    #     img = mmcv.bgr2rgb(img)
    #
    # if visualizer is not None:
    #     visualizer.add_datasample(
    #         'result',
    #         img,
    #         data_sample=data_samples,
    #         draw_gt=False,
    #         draw_heatmap=args.draw_heatmap,
    #         draw_bbox=args.draw_bbox,
    #         show_kpt_idx=args.show_kpt_idx,
    #         skeleton_style=args.skeleton_style,
    #         show=args.show,
    #         wait_time=show_interval,
    #         kpt_thr=args.kpt_thr)

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)


def inference(video_path, output_root, difference_bet_frame=3, smooth_sigma=20, num_circle=3, vis=True):
    pose_checkpoint = 'models/lsp_epoch_best_20.pth'
    pose_config = 'config/rtmpose-m_8xb256-420e_lsp-256x192.py'
    det_config = "config/rtmdet_m_640-8xb32_coco-person.py"
    det_checkpoint = "models/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
    detector = init_detector(det_config, det_checkpoint, device="cuda:0")
    pose_estimator = init_pose_estimator(
        pose_config,
        pose_checkpoint,
        device='cuda:0',
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=False))))

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
                pred_instances = process_one_image(frame, detector,
                                                   pose_estimator)

                csv_info = f"{frame_idx},"
                instances_info = split_instances(pred_instances)[0]
                kpts = instances_info["keypoints"]
                kpts_scores = instances_info["keypoint_scores"]

                for idx, kpt in enumerate(kpts):
                    csv_info += str(kpt[0]) + "," + str(kpt[1]) + "," + str(kpts_scores[idx]) + ","

                csv_info += "\n"
                output_csv.write(csv_info)
        else:
            break

        frame_idx += 1

    output_csv.close()
    progress_bar.close()
    cap.release()

    if vis:
        overwrite_video(video_path, output_root,
                        f"{os.path.join(output_root, os.path.splitext(os.path.basename(video_path))[0])}.csv",
                        diff=difference_bet_frame)
        out_video = f"{os.path.join(output_root, os.path.splitext(os.path.basename(video_path))[0])}.mp4"
        os.system(f"ffmpeg -i {out_video} -c:v libx264 -c:a aac {out_video.replace('.mp4', '_final.mp4')}")
        os.system(f"mv -f {out_video.replace('.mp4', '_final.mp4')} {out_video}")
    try:
        analysis_result = steps_analysis.main_analysis(
            f"{os.path.join(output_root, os.path.splitext(os.path.basename(video_path))[0])}.csv",
            smooth_sigma=smooth_sigma, num_circle=num_circle)
    except Exception as e:
        print(e)
        return None

    return analysis_result


if __name__ == '__main__':
    video_path = "/data/videos/original/1-72_raw.MP4-9a92c6db.mp4"
    output_root = "/data/videos/inference/"
    inference(video_path, output_root, difference_bet_frame=1, vis=True)
    # overwrite_video(video_path, output_root, "/data/videos/inference/1-72_raw.MP4-9a92c6db.csv", diff=1)
