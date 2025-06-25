import os
import sys
import cv2
import numpy as np
import csv
import json
import matplotlib.pyplot as plt
from ultralytics import YOLO
from keypoints_processing import caculate_output
from tqdm import tqdm
import requests
from collections import defaultdict
from tracker.byte_tracker import BYTETracker
from types import SimpleNamespace

REF_HEIGHT = 720
backend_host = os.getenv("BACKEND_HOST", "127.0.0.1")
# 给文件添加后缀
def add_suffix(filename, key):
    """添加后缀到文件名（保留扩展名）"""
    # 处理空文件名
    if not filename:
        return f"_{key}" if key else ""
    
    # 处理隐藏文件（以点开头的文件）
    if filename.startswith('.'):
        return f"{filename}_{key}"
    
    # 主要处理逻辑
    parts = filename.rsplit('.', 1)
    if len(parts) > 1:
        return f"{parts[0]}_{key}.{parts[1]}"
    else:
        return f"{filename}_{key}"
# 更新进度函数
def update_progress(action_id, current, total, phase=1):
    """更新进度，phase=1表示视频处理阶段，phase=2表示渲染阶段"""
    try:
        # 第一阶段占50%，第二阶段占50%
        progress = (phase-1)*0.5 + 0.5*(current/total)
        requests.post(
            f"http://{backend_host}:8000/api/v1/actions/update_action_progress",
            json={
                "action_id": action_id, 
                "progress": f"{round(progress * 100)}%"
            },
            timeout=1  # 添加超时
        )
    except Exception as e:
        #pass
        print(f"更新进度失败: {str(e)}")
# 验证四点是否构成一个有效的矩形
def validate_rectangle(points, min_distance=50):
    # 计算所有点之间的最小距离
    from scipy.spatial import distance
    dist_matrix = distance.cdist(points, points, 'euclidean')
    np.fill_diagonal(dist_matrix, np.inf)  # 忽略对角线
    min_dist = np.min(dist_matrix)
    
    # 判断是否存在两点过近
    if min_dist < min_distance:
        return False
    return True
# 从文件中读取特征点    
def get_featurepoints3():
    file_path = './points.txt'
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            points = [tuple(map(int, line.strip().split(','))) for line in lines]
            bottom_left, top_left, top_right, bottom_right = points
            sorted_points = [bottom_left, top_left, top_right, bottom_right]
            if len(points) == 4:
                return True, sorted_points
            else:
                print(f"Expected 4 points, but found {len(points)} in {file_path}")
                return False, []
    except Exception as e:
        print(f"Failed to read points from {file_path}: {str(e)}")
        return False, []
# 这个方法暂时被get_featurepoints3替代
def get_featurepoints2(frame):
    #debug_dir = "data/debug"
    #os.makedirs(debug_dir, exist_ok=True)
    
    # 保存原始图像
    #cv2.imwrite(f"{debug_dir}/0_original.jpg", frame)
    
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #cv2.imwrite(f"{debug_dir}/1_hsv.jpg", hsv)
    
    # 定义红色范围
    lower_red1 = np.array([0, 120, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # 创建红色掩膜
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    #cv2.imwrite(f"{debug_dir}/2_red_mask.jpg", red_mask)
    
    # 形态学处理
    kernel_size = int(max(frame.shape[:2]) / 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    processed = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
    #cv2.imwrite(f"{debug_dir}/3_processed.jpg", processed)
    
    # 查找轮廓
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_debug = frame.copy()
    #cv2.drawContours(contour_debug, contours, -1, (0,255,0), 2)
    #cv2.imwrite(f"{debug_dir}/4_all_contours.jpg", contour_debug)
    
    # 椭圆检测参数
    min_area = 400
    max_area = 8000
    candidates = []
    failure_reasons = []
    
    for i, cnt in enumerate(contours):
        cnt_debug = frame.copy()
        cv2.drawContours(cnt_debug, [cnt], -1, (0,255,0), 2)
        
        if len(cnt) < 5:
            # failure_reasons.append(f"Contour {i}: Not enough points ({len(cnt)} < 5)")
            continue
        
        # 椭圆拟合
        try:
            ellipse = cv2.fitEllipse(cnt)
        except:
            #failure_reasons.append(f"Contour {i}: Ellipse fitting failed")
            continue
            
        (x, y), (ma, MA), angle = ellipse
        cv2.ellipse(cnt_debug, ellipse, (255,0,0), 2)
        
        # 面积筛选
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            #failure_reasons.append(f"Contour {i}: Area too small ({area:.1f} < {min_area:.1f})")
            #cv2.imwrite(f"{debug_dir}/contour_{i}_area_fail.jpg", cnt_debug)
            continue
            
        # 凸性缺陷检测
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = float(area)/hull_area
        if solidity < 0.85:  # 排除凹陷形状
            #failure_reasons.append(f"Contour {i}: hull too low ({solidity:.2f} < 0.85")
            #cv2.imwrite(f"{debug_dir}/contour_{i}_hull_fail.jpg", cnt_debug)
            continue

        # 轴长比例筛选
        axis_ratio = min(ma, MA) / max(ma, MA)
        if axis_ratio < 0.2:
            #failure_reasons.append(f"Contour {i}: Axis ratio too low ({axis_ratio:.2f} < 0.5)")
            #cv2.imwrite(f"{debug_dir}/contour_{i}_ratio_fail.jpg", cnt_debug)
            continue
            
        # 圆度检测（周长^2与面积的比例）
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        if circularity < 0.5:  # 圆度阈值
            #failure_reasons.append(f"Contour {i}: Circularity too low ({circularity:.2f} < 0.6)")
            continue

        # 颜色验证
        roi_size = int(max(ma, MA)/2)
        x1 = max(0, int(x)-roi_size)
        y1 = max(0, int(y)-roi_size)
        x2 = min(frame.shape[1], int(x)+roi_size)
        y2 = min(frame.shape[0], int(y)+roi_size)
        roi = red_mask[y1:y2, x1:x2]
        
        red_percent = np.mean(roi)/255
        if red_percent < 0.2:
            #failure_reasons.append(f"Contour {i}: Red percentage low ({red_percent:.1%} < 40%)")
            #cv2.imwrite(f"{debug_dir}/contour_{i}_color_fail.jpg", cnt_debug)
            continue
            
        # 通过所有筛选
        candidates.append({
            'pos': (int(x), int(y)),
            'area': area,
            'axes': (ma, MA),
            'circularity': circularity
        })
        #cv2.imwrite(f"{debug_dir}/contour_{i}_passed.jpg", cnt_debug)
    
    # 候选点筛选
    debug_info = {
        'total_contours': len(contours),
        'min_area': min_area,
        'candidates_count': len(candidates),
        'failure_reasons': failure_reasons
    }
    '''
    with open(f"{debug_dir}/debug_info.json", "w") as f:
        json.dump(debug_info, f, indent=2)
    '''
    # 筛选四个最佳候选点
    if len(candidates) < 4:
        print(f"候选点不足（需要4个，找到{len(candidates)}个）")
        print("失败原因：")
        print("\n".join(failure_reasons))
        return False, []
    
    # 按面积+圆度排序并取前四个
    candidates.sort(key=lambda x: x['area'] * x['circularity'], reverse=True)
    selected = candidates[:4]
    points = np.array([s['pos'] for s in selected])
    
    # 几何验证
    if not validate_rectangle(points):
        print("几何验证失败，四点不符合四边形要求")
        return False, []
    
    # 排序点序
    try:
        # 按x坐标排序，将点分成左两组和右两组
        sorted_by_x = sorted(points, key=lambda p: p[0])
        left_points = sorted_by_x[:2]  # 左边两个点
        right_points = sorted_by_x[2:]  # 右边两个点

        # 左边点按y从小到大排序 -> 左上(y小)、左下(y大)
        left_sorted = sorted(left_points, key=lambda p: p[1])
        # 右边点按y从小到大排序 -> 右上(y小)、右下(y大)
        right_sorted = sorted(right_points, key=lambda p: p[1])

        # 组合顺序：左下、左上、右上、右下
        top_left, bottom_left = left_sorted  # 解包时交换顺序
        top_right, bottom_right = right_sorted
        sorted_points = [bottom_left, top_left, top_right, bottom_right]
        print(f"Sorted points: {sorted_points}")
    except Exception as e:
        print(f"点排序失败：{str(e)}")
        return False, []
    
    # 最终调试图像
    debug_frame = frame.copy()
    print(f"YOLO detected {len(sorted_points)} points: {sorted_points}")

    for pt in sorted_points:
        cv2.circle(debug_frame, tuple(pt), 10, (0,255,0), -1)
    # cv2.imwrite(f"data/1-calibration_screenshot.png", debug_frame)
    
    return True, sorted_points
# 用来计算透视矩阵的函数
def get_perspective_matrix(frame, point_model, frame_count, orig_frame, fps=30.0):
    # if frame_count < 3 * fps:
    if 1:
        results = point_model(frame)
        points = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                points.append((center_x, center_y))
        
        print(f"Frame {frame_count}: YOLO detected {len(points)} points: {points}")
        
        if len(points) == 4:
            # 按x坐标排序，将点分成左两组和右两组
            sorted_by_x = sorted(points, key=lambda p: p[0])
            left_points = sorted_by_x[:2]  # 左边两个点
            right_points = sorted_by_x[2:]  # 右边两个点

            # 左边点按y从小到大排序 -> 左上(y小)、左下(y大)
            left_sorted = sorted(left_points, key=lambda p: p[1])
            # 右边点按y从小到大排序 -> 右上(y小)、右下(y大)
            right_sorted = sorted(right_points, key=lambda p: p[1])

            # 组合顺序：左下、左上、右上、右下
            top_left, bottom_left = left_sorted  # 解包时交换顺序
            top_right, bottom_right = right_sorted
            sorted_points = [bottom_left, top_left, top_right, bottom_right]
            print(f"Sorted points: {sorted_points}")
            # 几何验证
            if validate_rectangle(sorted_points):
                print(f"Valid rectangle points: {sorted_points}")
            else:
                print("Invalid rectangle detected, using fallback")
                return None, None
                ret, sorted_points = get_featurepoints3()
                # ret, sorted_points = get_featurepoints2(orig_frame)
                if not ret:
                    return None, None
            # 绘制标记连线并保存截图
            # for i in range(4):
                # cv2.line(frame, sorted_points[i], sorted_points[(i+1)%4], (0, 255, 255), 2)
            # cv2.imwrite("data/1-calibration_screenshot.png", frame)
        else:
            return None, None

            ret, sorted_points = get_featurepoints3()
            # ret, sorted_points = get_featurepoints2(orig_frame)
            # print(f"Frame {frame_count}: OpenCV fallback - {'success' if ret else 'failed'}")
            if not ret:
                return None, None
        
        max_width, max_height = 1280, 720
        expand_pixels = 60
        dst_points = np.array([
            [expand_pixels, max_height - 1 - expand_pixels],
            [expand_pixels, expand_pixels],
            [max_width - 1 - expand_pixels, expand_pixels],
            [max_width - 1 - expand_pixels, max_height - 1 - expand_pixels]
        ], dtype='float32')
        
        src_points = np.array(sorted_points, dtype='float32')
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        print(f"Perspective matrix calculated at frame {frame_count}!")
        return M, sorted_points
    return None, None

    '''
def main(input_video_file, out_video_file, out_json_file, out_warped_video_file=None):
    action_id = "1"
    '''
def main(action_id, input_video_file, out_video_file, out_json_file, out_warped_video_file=None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pose_model_path = os.path.join(current_dir, 'yolov8x-pose.pt')
    point_model_path = os.path.join(current_dir, 'yolov8s-point.pt')
    
    pose_model = YOLO(pose_model_path)
    point_model = YOLO(point_model_path)
    
    result = 1

    # 读取第一帧判断镜像需求
    cap = cv2.VideoCapture(input_video_file)
    if not cap.isOpened():
        print(f"Failed to open input video: {input_video_file}")
        return
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read first frame for mirror check")
        return
    # 使用YOLO模型检测第一帧的关键点
    # 这一步的目的是用来判断视频是否需要镜像处理
    results = pose_model.predict(first_frame, conf=0.8, iou=0.2, verbose=False)[0]
    keypoints_ = results.keypoints.xy.cpu().numpy()[0]
    mirror_flag = False
    if len(keypoints_) >= 17:
        right_hip_x = keypoints_[12][0]
        frame_width = first_frame.shape[1]
        if right_hip_x > frame_width / 2:
            mirror_flag = True
    cap.release()

    cap = cv2.VideoCapture(input_video_file)  # 重新打开视频
    mid_video_file = 'mid_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0
    original_fps = round(cap.get(cv2.CAP_PROP_FPS))
    target_fps = 30.0
    # 定一个中间视频文件
    video_writer = cv2.VideoWriter(mid_video_file, fourcc, target_fps, (1280, 720))
    
    if not cap.isOpened():
        print(f"Failed to open input video: {input_video_file}")
        return
    
    # 在此处定一个跟踪器
    # tracker_args = {
    #     'track_buffer': fps,  # 跟踪缓冲区大小
    #     'match_thresh': 0.8,  # 匹配阈值
    #     'track_thresh': 0.5,  # 跟踪阈值
    #     'aspect_ratio_thresh': 1.6,  # 宽高比阈值
    #     'min_box_area': 10,  # 最小边界框面积
    #     'mot20': False,  # 是否使用MOT20数据集的参数
    # }
    tracker_args = SimpleNamespace(
        track_buffer=fps, # 跟踪缓冲区大小
        match_thresh=0.7, # 匹配阈值
        track_thresh=0.5, # 跟踪阈值
        aspect_ratio_thresh=1.6, # 宽高比阈值
        min_box_area=10, # 最小边界框面积
        mot20=False # 是否使用MOT20数据集的参数
    )
    tracker = BYTETracker(tracker_args, frame_rate=fps)
    
    # 获取视频总帧数并初始化进度条
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_total = int(total_frames * (target_fps / original_fps))
    pbar_video = tqdm(total=processed_total, desc="Processing video frames")  # 新增进度条

    # 添加ID计数器
    id_frame_counter = defaultdict(int)

    if not video_writer.isOpened():
        # print(f"Failed to create output video: {mid_video_file}")
        return
    
    warped_video_writer = None
    if out_warped_video_file:
        warped_video_writer = cv2.VideoWriter(out_warped_video_file, fourcc, fps, (1280, 720))
        if not warped_video_writer.isOpened():
            print(f"Failed to create warped video: {out_warped_video_file}")
            return
    # 数据需要封装成多人，使用一个二位数组来表示，第一个维度表示id，第二个维度表示数据
    # 第一个维度的数据后面使用bytetrack的追踪id来替代    
    # data = {}  # 原始坐标
    data = defaultdict(list)
    M = None
    key_points = []
    frame_count = 0
    first_valid_M = None
    first_valid_key_points = None
    sample_interval = original_fps / target_fps
    next_sample = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count >= int(next_sample):
            # 镜像处理
            if mirror_flag:
                frame = cv2.flip(frame, 1)
            orig_frame = frame.copy()
            # 计算比例因子 (基于参考高度720)
            current_height = orig_frame.shape[0]
            scale = current_height / REF_HEIGHT
            last_orig_frame = orig_frame
            # 使用模型时不再需要指定 half=True，因为模型本身已经是FP16了
            current_M, current_key_points = get_perspective_matrix(frame, point_model, frame_count, orig_frame, fps)
            if current_M is not None and current_key_points:
                first_valid_M = current_M
                first_valid_key_points = current_key_points
                break
            next_sample += sample_interval
        frame_count += 1
    cap.release()

    # ========== 正式采集数据 ========== 
    cap = cv2.VideoCapture(input_video_file)
    frame_count = 0
    last_orig_frame = None
    last_M = None
    last_key_points = None
    # 强制初始化，避免UnboundLocalError
    last_valid_M = first_valid_M
    last_valid_key_points = first_valid_key_points
    threshold = 200
    processed_frames = 0
    sample_interval = original_fps / target_fps
    next_sample = 0
    # 采样后帧数进度条
    pbar_video = tqdm(total=processed_total, desc="Processing video frames")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count >= int(next_sample):
            # 镜像处理
            if mirror_flag:
                frame = cv2.flip(frame, 1)
            orig_frame = frame.copy()
            # 计算比例因子 (基于参考高度720)
            current_height = orig_frame.shape[0]
            scale = current_height / REF_HEIGHT
            last_orig_frame = orig_frame
            # 获取当前帧的透视矩阵和关键点
            current_M, current_key_points = get_perspective_matrix(frame, point_model, frame_count, orig_frame, fps)
            # 关键点稳定性处理
            if current_M is not None and current_key_points:
                if last_valid_key_points is not None:
                    total_diff = sum(
                        ((cur[0]-last[0])**2 + (cur[1]-last[1])**2)**0.5 
                        for cur, last in zip(current_key_points, last_valid_key_points)
                    )
                    if total_diff > threshold:
                        M = last_valid_M
                        key_points = last_valid_key_points
                        print(f"Frame {frame_count}: Jump detected ({total_diff:.1f}px), using previous points")
                    else:
                        M = current_M
                        key_points = current_key_points
                        last_valid_M = current_M
                        last_valid_key_points = [tuple(pt) for pt in current_key_points]
                else:
                    M = current_M
                    key_points = current_key_points
                    last_valid_M = current_M
                    last_valid_key_points = [tuple(pt) for pt in current_key_points]
            else:
                if last_valid_M is not None:
                    M = last_valid_M
                    key_points = last_valid_key_points
                else:
                    M = first_valid_M
                    key_points = first_valid_key_points
            # 修改2: 使用模型时不再需要指定 half=True
            results = pose_model.predict(orig_frame, conf=0.5, iou=0.5, imgsz=640, verbose=False)[0]

            # 提取检测信息
            detections = []
            track_cls_ids = []     # 类别ID列表
            track_keypoints = []   # 关键点数据列表
            # 迭代识别结果
            for idx, box in enumerate(results.boxes):
                # 获取边界框坐标 (xyxy格式: x_min, y_min, x_max, y_max)
                bbox = box.xyxy[0].tolist()
                # 获取置信度
                confidence = box.conf[0].item()
                # 获取类别ID和名称
                cls_id = int(box.cls[0].item())
                # cls_name = results.names[cls_id]
                keypoints_ = results.keypoints[idx].xy[0].cpu().numpy()
                # 添加到检测列表（ByteTrack 需要的格式）
                detections.append([bbox[0], bbox[1], bbox[2], bbox[3], confidence])
                # 获取类别ID
                track_cls_ids.append(cls_id)
                
                # 获取关键点数据
                keypoints_ = results.keypoints[idx].xy[0].cpu().numpy()
                track_keypoints.append(keypoints_)
            if not detections:
                # 如果没有检测到目标，跳过当前帧
                pbar_video.update(1)
                continue
            # 转换为NumPy数组
            detections = np.array(detections)
            track_cls_ids = np.array(track_cls_ids)
            track_keypoints = np.array(track_keypoints)
            tracks = tracker.update((detections,track_cls_ids,track_keypoints))
            for track in tracks:
                track_id = track.track_id
                id_frame_counter[track_id] += 1
            # 确定当前帧数最多的ID（在整个视频中）
            if id_frame_counter:
                max_id = max(id_frame_counter, key=id_frame_counter.get)
            else:
                max_id = None

            # 现在只绘制max_id
            for track in tracks:
                track_id = track.track_id

                # 只绘制帧数最多的ID
                if track_id == max_id:
                    bbox = track.tlbr  # 边界框 [x1, y1, x2, y2]
                    """
                    # 在原始帧上绘制边界框和跟踪 ID
                    box_thickness = max(1, int(2 * scale))
                    if mirror_flag:
                        frame = cv2.flip(frame, 1)
                        orig_frame = frame
                        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        cv2.rectangle(orig_frame, (original_width - int(bbox[0]), int(bbox[1])),
                                    (original_width - int(bbox[2]), int(bbox[3])), (0, 255, 0), box_thickness)
                        
                        text_to_draw = f"ID: {track_id}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1 * scale
                        # font_thickness = 2
                        font_thickness = max(1, int(2 * scale))
                        text_color = (0, 0, 0) # BGR Black

                        # 先获取文本尺寸，以便精确定位
                        (text_w, text_h), baseline = cv2.getTextSize(text_to_draw, font, font_scale, font_thickness)
                        padding = 10
                        # 正常视频，标签画在框的左边
                        text_x = original_width - int(bbox[2]) - text_w - padding
                        
                        # y坐标: 在框顶部向下的1/5 (20%) 高度处
                        text_y = int(bbox[1] + (bbox[3] - bbox[1]) * 0.2)
                        text_pos = (text_x, text_y)
                        cv2.putText(orig_frame, text_to_draw, text_pos, font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                        frame = cv2.flip(frame, 1)
                        orig_frame = frame
                    else:
                        cv2.rectangle(orig_frame, (int(bbox[0]), int(bbox[1])),
                                    (int(bbox[2]), int(bbox[3])), (0, 255, 0), box_thickness)
                        
                        text_to_draw = f"ID: {track_id}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1 * scale
                        # font_thickness = 2
                        font_thickness = max(1, int(2 * scale))
                        text_color = (0, 0, 0) # BGR Black

                        # 先获取文本尺寸，以便精确定位
                        (text_w, text_h), baseline = cv2.getTextSize(text_to_draw, font, font_scale, font_thickness)
                        padding = 10
                        # 正常视频，标签画在框的左边
                        text_x = int(bbox[0]) - text_w - padding
                        
                        # y坐标: 在框顶部向下的1/5 (20%) 高度处
                        text_y = int(bbox[1] + (bbox[3] - bbox[1]) * 0.2)
                        text_pos = (text_x, text_y)
                        cv2.putText(orig_frame, text_to_draw, text_pos, font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                    """
                    """
                    # 计算标签位置:
                    padding = 10
                    if mirror_flag:
                        frame = cv2.flip(frame, 1)

                        # 如果是镜像视频，标签应该画在框的右边，这样翻转后视觉上才在左边
                        text_x = int(bbox[2]) + padding
                    else:
                        # 正常视频，标签画在框的左边
                        text_x = int(bbox[0]) - text_w - padding
                    
                    # y坐标: 在框顶部向下的1/5 (20%) 高度处
                    text_y = int(bbox[1] + (bbox[3] - bbox[1]) * 0.2)
                    text_pos = (text_x, text_y)

                    if mirror_flag:
                        # 为了防止文字镜像，我们先在一个临时画布上绘制文字，
                        # 然后水平翻转这个画布，再把它叠加到主画面上。
                        
                        # 1. 获取文字尺寸
                        (text_w, text_h), baseline = cv2.getTextSize(text_to_draw, font, font_scale, font_thickness)

                        # 2. 创建一个足够大的临时画布 (黑色背景)
                        text_canvas = np.zeros((text_h + baseline, text_w, 3), np.uint8)

                        # 3. 在画布上绘制【白色】文字，以便生成正确的掩码
                        cv2.putText(text_canvas, text_to_draw, (0, text_h), font, font_scale, (255, 255, 255), max(1, int(font_thickness * scale)), cv2.LINE_AA)
                        
                        # 4. 水平翻转文字画布
                        text_canvas = cv2.flip(text_canvas, 1)

                        # 5. 计算要粘贴到主图像上的位置
                        x, y_baseline = text_pos
                        y_top = y_baseline - text_h # 文字框的顶部y坐标
                        
                        # 确保ROI在图像范围内
                        if (y_top >= 0 and x >= 0 and
                            y_top + text_canvas.shape[0] <= orig_frame.shape[0] and
                            x + text_canvas.shape[1] <= orig_frame.shape[1]):
                            
                            roi = orig_frame[y_top : y_top + text_canvas.shape[0], x : x + text_canvas.shape[1]]
                            
                            # 6. 创建掩码，现在可以正确选择出白色文字了
                            text_gray = cv2.cvtColor(text_canvas, cv2.COLOR_BGR2GRAY)
                            _, mask = cv2.threshold(text_gray, 0, 255, cv2.THRESH_BINARY)
                            mask_inv = cv2.bitwise_not(mask)
                            
                            # 7. 融合：
                            #   - 从ROI中抠出文字区域，得到背景
                            bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                            
                            #   - 创建一个纯色的前景，颜色为最终想要的文字颜色
                            fg_colored = np.full(roi.shape, text_color, dtype=np.uint8)
                            
                            #   - 只在掩码区域（文字区域）应用这个颜色
                            fg = cv2.bitwise_and(fg_colored, fg_colored, mask=mask)
                            
                            #   - 相加得到最终结果
                            dst = cv2.add(bg, fg)
                            orig_frame[y_top : y_top + text_canvas.shape[0], x : x + text_canvas.shape[1]] = dst
                        else: # ROI 超出边界，使用简单绘制 (这种情况下文字会被镜像)
                            cv2.putText(orig_frame, text_to_draw, text_pos, font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                    else:
                        cv2.putText(orig_frame, text_to_draw, text_pos, font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                    """
                    cls_id = track.cls_id  # 类别ID
                    keypoints_ = track.keypoints  # 关键点数据
                    # for idx, box in enumerate(results.boxes):   
                    # cls_name = results.names[cls_id]
                    # keypoints_ = results.keypoints[idx].xy[0].cpu().numpy()
                    keypoints = np.zeros((8, 2), dtype=np.float32)  # 左肩、右肩、左髋、右髋、左膝、右膝、左脚踝、右脚踝
                    if len(keypoints_) >= 17:  # COCO关键点模型
                        keypoints[0] = keypoints_[5]   # 左肩
                        keypoints[1] = keypoints_[6]   # 右肩
                        keypoints[2] = keypoints_[11]  # 左髋
                        keypoints[3] = keypoints_[12]  # 右髋
                        keypoints[4] = keypoints_[13]  # 左膝
                        keypoints[5] = keypoints_[14]  # 右膝
                        keypoints[6] = keypoints_[15]  # 左脚踝
                        keypoints[7] = keypoints_[16]  # 右脚踝
                        
                    else:
                        pass  # 处理检测失败的情况
                    
                    # 绘制地面红点连线
                    if M is not None and key_points: # and frame_count < 3 * fps
                        marker_radius = max(1, int(5 * scale))
                        line_thickness = max(1, int(2 * scale))
                        for i, pt in enumerate(key_points):
                            cv2.circle(orig_frame, pt, marker_radius, (0, 255, 0), -1)
                            # cv2.putText(orig_frame, str(i+1), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.line(orig_frame, key_points[0], key_points[1], (0, 255, 255), line_thickness)
                        cv2.line(orig_frame, key_points[1], key_points[2], (0, 255, 255), line_thickness)
                        cv2.line(orig_frame, key_points[2], key_points[3], (0, 255, 255), line_thickness)
                        cv2.line(orig_frame, key_points[3], key_points[0], (0, 255, 255), line_thickness)
                    
                    if len(keypoints_) > 0:
                        keypoint_radius = max(1, int(8 * scale))
                        bone_thickness = max(1, int(4 * scale))
                        for i, (x, y) in enumerate(keypoints_):
                            if i in [5, 6, 11, 12, 13, 14, 15, 16]:
                                color = (255, 0, 0) if i in [5, 11, 13, 15] else (0, 0, 255) if i in [6, 12, 14, 16] else (255, 0, 0)
                                cv2.circle(orig_frame, (int(x), int(y)), keypoint_radius, color, -1)
                        if len(keypoints_) >= 17:
                            if np.all(keypoints_[5]) and np.all(keypoints_[6]):
                                cv2.line(orig_frame, (int(keypoints_[5, 0]), int(keypoints_[5, 1])), 
                                    (int(keypoints_[6, 0]), int(keypoints_[6, 1])), (0, 255, 0), bone_thickness)
                            if np.all(keypoints_[11]) and np.all(keypoints_[12]):
                                cv2.line(orig_frame, (int(keypoints_[11, 0]), int(keypoints_[11, 1])), 
                                    (int(keypoints_[12, 0]), int(keypoints_[12, 1])), (0, 255, 0), bone_thickness)
                            if np.all(keypoints_[5]) and np.all(keypoints_[11]):
                                cv2.line(orig_frame, (int(keypoints_[5, 0]), int(keypoints_[5, 1])), 
                                    (int(keypoints_[11, 0]), int(keypoints_[11, 1])), (0, 255, 0), bone_thickness)
                            if np.all(keypoints_[11]) and np.all(keypoints_[13]):
                                cv2.line(orig_frame, (int(keypoints_[11, 0]), int(keypoints_[11, 1])),
                                    (int(keypoints_[13, 0]), int(keypoints_[13, 1])), (255, 0, 0), bone_thickness)
                            if np.all(keypoints_[13]) and np.all(keypoints_[15]):
                                cv2.line(orig_frame, (int(keypoints_[13, 0]), int(keypoints_[13, 1])), 
                                    (int(keypoints_[15, 0]), int(keypoints_[15, 1])), (255, 0, 0), bone_thickness)
                            if np.all(keypoints_[6]) and np.all(keypoints_[12]):
                                cv2.line(orig_frame, (int(keypoints_[6, 0]), int(keypoints_[6, 1])),
                                        (int(keypoints_[12, 0]), int(keypoints_[12, 1])), (0, 255, 0), bone_thickness)
                            if np.all(keypoints_[12]) and np.all(keypoints_[14]):
                                cv2.line(orig_frame, (int(keypoints_[12, 0]), int(keypoints_[12, 1])),
                                        (int(keypoints_[14, 0]), int(keypoints_[14, 1])), (0, 0, 255), bone_thickness)
                            if np.all(keypoints_[14]) and np.all(keypoints_[16]):
                                cv2.line(orig_frame, (int(keypoints_[14, 0]), int(keypoints_[14, 1])), 
                                    (int(keypoints_[16, 0]), int(keypoints_[16, 1])), (0, 0, 255), bone_thickness)
                    data[track_id].append(keypoints)
            
            # video_writer.write(orig_frame)
            resized_orig_frame = cv2.resize(orig_frame, (1280, 720))
            # 保存原始帧到中间视频
            video_writer.write(resized_orig_frame)
            pbar_video.update(1)
            # 每%10更新一次进度条
            if processed_frames % 10 == 0:
                update_progress(action_id, processed_frames, processed_total, phase=1)
            processed_frames += 1
            next_sample += sample_interval
        frame_count += 1
    cap.release()
    video_writer.release()
    if warped_video_writer:
        warped_video_writer.release()
    cv2.destroyAllWindows()
    
    if M is None or not key_points:
        print("No perspective matrix calculated, check video or models!")
    else:
        # 过滤掉左右脚踝坐标中任一值为0的时间点数据
        filtered_data = defaultdict(list)
        for key, value in data.items():
            for idx, key_points_temp in enumerate(value):
                knee_and_ankle_points = key_points_temp[4:8]
                if np.all(knee_and_ankle_points):
                    filtered_data[key].append(key_points_temp)
        
        id_frame_counts = {k: len(v) for k, v in filtered_data.items()}

        # 只保留过滤后帧数最多的ID
        if filtered_data:
            print("各ID帧数统计：")
            for k, v in filtered_data.items():
                print(f"ID {k}: {len(v)} 帧")
            max_id = max(filtered_data, key=lambda k: len(filtered_data[k]))
            data = {max_id: filtered_data[max_id]}
        else:
            data = {}
        # 保存最终确定的max_id用于渲染
        final_max_id = max_id

        # 保存左右脚脚踝关键点坐标（只保存左右脚踝）
    
        print(f"Video processing completed. Total frames: {frame_count}")

        
         # 渲染参数到视频
        def render_parameters_to_video(input_video_path, output_video_path, scaled_data, smoothed_speed, gait_result, mirror_flag, id_counts=None):
            merged_data = []
            
            # 获取所有时间戳（使用缩放数据的时间轴）
            times = [d['time'] for d in scaled_data] 
            
            # 构建速度字典（提高查询效率）
            speed_dict = {s['time']: s for s in smoothed_speed}

            for d in scaled_data:
                time = d['time']
                speed_entry = speed_dict.get(time, {'right_speed': 0})
                
                merged_entry = {
                    'time': time,
                    'right_ankle_x': d['right_ankle'][0],
                    'right_ankle_y': d['right_ankle'][1],
                    'right_speed': speed_entry['right_speed']
                }
                merged_data.append(merged_entry)
    
            merged_data.sort(key=lambda x: x['time'])
            
            # 打开输入视频
            cap = cv2.VideoCapture(input_video_path)

            total_render_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            pbar_render = tqdm(total=total_render_frames, desc="Rendering parameters")  # 新增进度条

            fps = cap.get(cv2.CAP_PROP_FPS)
            # fps = 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 创建输出视频，fps用target_fps
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, target_fps, (width, height))

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if mirror_flag:
                    frame = cv2.flip(frame, 1)
                current_time = frame_count / fps
                frame_count += 1

                # 查找当前帧对应的步态信息
                current_step_info = None
                current_step_number = 0
                total_steps = sum(len(stage['steps_info']) for stage in gait_result)
                
                for stage in gait_result:
                    for idx, step in enumerate(stage['steps_info']):
                        if step['start_frame'] <= frame_count and step['end_frame'] >= frame_count:
                            current_step_info = step
                            current_step_number = idx + 1
                            break
                    if current_step_info:
                        break

                # 查找最近的时间点
                if not merged_data:
                    closest_idx = 0
                else:
                    times_array = np.array(times)
                    closest_idx = np.argmin(np.abs(times_array - current_time))
                    closest_time = merged_data[closest_idx]['time']

                # 获取参数
                if merged_data and abs(closest_time - current_time) < 1/fps:
                    params = merged_data[closest_idx]
                    right_ankle_x = params['right_ankle_x']
                    right_ankle_y = params['right_ankle_y']
                    right_speed = params['right_speed']
                else:
                    right_ankle_x = right_ankle_y = right_speed = 0.0

                # 在右上方渲染参数
                text_y = 50
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2

                background_color = (255, 255, 255)  # 白色
                text_color = (0, 0, 0)  # 黑色文字

                # 第一行文字
                text = f"Time: {current_time:.2f}s"
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_width, text_height = text_size
                text_x = width - 300

                # 绘制背景矩形
                cv2.rectangle(frame, (text_x - 10, text_y - text_height - 5), 
                            (text_x + text_width + 10, text_y + 5), 
                            background_color, -1)

                # 绘制文字
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness)

                # 第二行文字
                text_y += 40
                text = f"Frames: {frame_count:.2f}"
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_width, text_height = text_size

                # 绘制背景矩形
                cv2.rectangle(frame, (text_x - 10, text_y - text_height - 5), 
                            (text_x + text_width + 10, text_y + 5), 
                            background_color, -1)

                # 绘制文字
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness)
                
                """
                if id_counts:
                    sorted_ids = sorted(id_counts.keys())
                    for track_id in sorted_ids:
                        count = id_counts[track_id]
                        text_y += 40
                        id_text = f"Focused ID: {track_id}({count})"
                        
                        text_size, _ = cv2.getTextSize(id_text, font, font_scale, thickness)
                        text_width, text_height = text_size
                        
                        # 绘制背景矩形
                        cv2.rectangle(frame, (text_x - 10, text_y - text_height - 5), 
                                    (text_x + text_width + 10, text_y + 5), 
                                    background_color, -1)
                        # 绘制文字
                        cv2.putText(frame, id_text, (text_x, text_y), font, font_scale, text_color, thickness)
                """

                # 第三行文字
                text_y += 40
                text = f"Right Ankle X: {right_ankle_x:.2f}m"
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_width, text_height = text_size

                # 绘制背景矩形
                cv2.rectangle(frame, (text_x - 10, text_y - text_height - 5), 
                            (text_x + text_width + 10, text_y + 5), 
                            background_color, -1)

                # 绘制文字
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness)

                # 第四行文字
                text_y += 40
                text = f"Right Ankle Y: {right_ankle_y:.2f}m"
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_width, text_height = text_size

                # 绘制背景矩形
                cv2.rectangle(frame, (text_x - 10, text_y - text_height - 5), 
                            (text_x + text_width + 10, text_y + 5), 
                            background_color, -1)

                # 绘制文字
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness)

                
                # 渲染步数和前腿信息
                if current_step_info:
                    # 绘制步数
                    text_y += 40
                    step_text = f"Step: {current_step_number}/{total_steps}"
                    text_size, _ = cv2.getTextSize(step_text, font, font_scale, thickness)
                    text_width, text_height = text_size
                    # 绘制背景矩形
                    cv2.rectangle(frame, (text_x - 10, text_y - text_height - 5), 
                                (text_x + text_width + 10, text_y + 5), 
                                background_color, -1)
                    # 绘制文字
                    cv2.putText(frame, step_text, (text_x, text_y), font, font_scale, text_color, thickness)
                    
                    # 绘制前腿信息
                    text_y += 40
                    leg_text = f"Swing Leg: {current_step_info['front_leg']}"
                    text_size, _ = cv2.getTextSize(leg_text, font, font_scale, thickness)
                    text_width, text_height = text_size
                    # 绘制背景矩形
                    cv2.rectangle(frame, (text_x - 10, text_y - text_height - 5), 
                                (text_x + text_width + 10, text_y + 5), 
                                background_color, -1)
                    # 绘制文字
                    cv2.putText(frame, leg_text, (text_x, text_y), font, font_scale, text_color, thickness)
                else:
                    # 绘制步数
                    text_y += 40
                    step_text = f"Step:"
                    text_size, _ = cv2.getTextSize(step_text, font, font_scale, thickness)
                    text_width, text_height = text_size
                    # 绘制背景矩形
                    cv2.rectangle(frame, (text_x - 10, text_y - text_height - 5), 
                                (text_x + text_width + 10, text_y + 5), 
                                background_color, -1)
                    # 绘制文字
                    cv2.putText(frame, step_text, (text_x, text_y), font, font_scale, text_color, thickness)
                    
                    # 绘制前腿信息
                    text_y += 40
                    leg_text = f"Swing Leg:"
                    text_size, _ = cv2.getTextSize(leg_text, font, font_scale, thickness)
                    text_width, text_height = text_size
                    # 绘制背景矩形
                    cv2.rectangle(frame, (text_x - 10, text_y - text_height - 5), 
                                (text_x + text_width + 10, text_y + 5), 
                                background_color, -1)
                    # 绘制文字
                    cv2.putText(frame, leg_text, (text_x, text_y), font, font_scale, text_color, thickness)




                out.write(frame)
                pbar_render.update(1)
                # 每%10更新一次进度条
                if frame_count % 10 == 0:  # 每10帧更新一次
                    update_progress(action_id, frame_count, total_render_frames, phase=2)
            pbar_render.close()
            cap.release()
            out.release()
            print(f"参数渲染完成，输出视频：{output_video_path}")

        # 调用计算输出
        for key, value in data.items():
            out_json_file_key = add_suffix(out_json_file, key)
            scaled_data, times, left_speed, right_speed, smoothed_data, result = caculate_output(key_points, np.array(value), out_json_file_key, M, fps=target_fps)
            if result is None:
                print(f"Error calculating output for key {key}, skipping...")
                continue
            smoothed_speed = [{'time': t, 'right_speed': rs} for t, rs in zip(times, right_speed)]
            
            if final_max_id is not None and final_max_id in id_frame_counts:
                final_id_count = {final_max_id: id_frame_counts[final_max_id]}
            else:
                final_id_count = None

            render_parameters_to_video(mid_video_file, out_video_file, scaled_data, smoothed_speed, result, mirror_flag, final_id_count)
            os.system(
                f"ffmpeg -y -i {out_video_file} -c:v libx264 -c:a aac -pix_fmt yuv420p {out_video_file.replace('.mp4', '_final.mp4')}")
            os.system(
                f"mv -f {out_video_file.replace('.mp4', '_final.mp4')} {out_video_file}")
        return result

if __name__ == "__main__":
    if len(sys.argv) not in [4, 5]:
        print("Usage: python inference.py input_video.mp4 out_video.mp4 out_json.json [out_warped_video.mp4]")
        sys.exit(-1)
    out_warped_video_file = sys.argv[4] if len(sys.argv) > 4 else None
    main(sys.argv[1], sys.argv[2], sys.argv[3], out_warped_video_file)