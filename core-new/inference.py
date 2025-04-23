import os
import sys
import cv2
import numpy as np
import csv
import json
import matplotlib.pyplot as plt
from ultralytics import YOLO
from check_point_v8 import get_featurepoints2, caculate_output
from tqdm import tqdm
import requests

backend_host = os.getenv("BACKEND_HOST", "127.0.0.1")

def update_progress(action_id, current, total, phase=1):
    """更新进度，phase=1表示视频处理阶段，phase=2表示渲染阶段"""
    try:
        # 第一阶段占50%，第二阶段占50%
        progress = (phase-1)*0.5 + 0.5*(current/total)
        requests.post(
            f"http://{backend_host}:8000/api/v1/actions/update_action_progress",
            json={
                "action_id": action_id, 
                "progress": str(round(progress, 2))
            },
            timeout=1  # 添加超时
        )
    except Exception as e:
        print(f"更新进度失败: {str(e)}")

def get_perspective_matrix(frame, point_model, frame_count, orig_frame, fps=24.0):
    if frame_count < 2 * fps:
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
            points = sorted(points, key=lambda x: (x[0], x[1]))
            bottom_left, top_left, top_right, bottom_right = points
            sorted_points = [bottom_left, top_left, top_right, bottom_right]
            print(f"Sorted points: {sorted_points}")
            
            # 绘制标记连线并保存截图
            for i in range(4):
                cv2.line(frame, sorted_points[i], sorted_points[(i+1)%4], (0, 255, 255), 2)
            cv2.imwrite("data/1-calibration_screenshot.png", frame)
        else:
            ret, sorted_points = get_featurepoints2(orig_frame)
            print(f"Frame {frame_count}: OpenCV fallback - {'success' if ret else 'failed'}")
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

def main(action_id, input_video_file, out_video_file, out_json_file, out_warped_video_file=None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pose_model_path = os.path.join(current_dir, 'yolov8x-pose.pt')
    point_model_path = os.path.join(current_dir, 'yolov8s-point.pt')
    
    pose_model = YOLO(pose_model_path)
    point_model = YOLO(point_model_path)
    
    mid_video_file = 'mid_video.mp4'
    cap = cv2.VideoCapture(input_video_file)
    if not cap.isOpened():
        print(f"Failed to open input video: {input_video_file}")
        return
    
    # 获取视频总帧数并初始化进度条
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar_video = tqdm(total=total_frames, desc="Processing video frames")  # 新增进度条

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 24.0
    video_writer = cv2.VideoWriter(mid_video_file, fourcc, fps, (1280, 720))
    
    if not video_writer.isOpened():
        # print(f"Failed to create output video: {mid_video_file}")
        pass
        return
    
    warped_video_writer = None
    if out_warped_video_file:
        warped_video_writer = cv2.VideoWriter(out_warped_video_file, fourcc, fps, (1280, 720))
        if not warped_video_writer.isOpened():
            print(f"Failed to create warped video: {out_warped_video_file}")
            return
        
    data = []  # 原始坐标
    M = None
    key_points = []
    frame_count = 0
    last_orig_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        orig_frame = frame.copy()
        last_orig_frame = orig_frame
        
        if M is None:
            M, key_points = get_perspective_matrix(frame, point_model, frame_count, orig_frame, fps)
        
        results = pose_model.predict(frame, conf=0.8, iou=0.2, verbose=False)[0]
        keypoints_ = results.keypoints.xy.cpu().numpy()[0]
        
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
        
        '''
        if M is not None and key_points:
            for i, pt in enumerate(key_points):
                cv2.circle(orig_frame, pt, 5, (0, 255, 0), -1)
                cv2.putText(orig_frame, str(i+1), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.line(orig_frame, key_points[0], key_points[1], (0, 255, 255), 2)
            cv2.line(orig_frame, key_points[1], key_points[2], (0, 255, 255), 2)
            cv2.line(orig_frame, key_points[2], key_points[3], (0, 255, 255), 2)
            cv2.line(orig_frame, key_points[3], key_points[0], (0, 255, 255), 2)
        '''
        
        if len(keypoints_) > 0:
            for i, (x, y) in enumerate(keypoints_):
                if i in [13, 14, 15, 16]:
                    color = (255, 0, 0) if i in [13, 15] else (0, 0, 255) if i in [14, 16] else (255, 0, 0)
                    cv2.circle(orig_frame, (int(x), int(y)), 8, color, -1)
            if len(keypoints_) > 13:
                cv2.line(orig_frame, (int(keypoints_[13, 0]), int(keypoints_[13, 1])), 
                        (int(keypoints_[15, 0]), int(keypoints_[15, 1])), (255, 0, 0), 4)
            if len(keypoints_) > 14:
                cv2.line(orig_frame, (int(keypoints_[14, 0]), int(keypoints_[14, 1])), 
                        (int(keypoints_[16, 0]), int(keypoints_[16, 1])), (0, 0, 255), 4)
            data.append(keypoints)
        
        # video_writer.write(orig_frame)
        resized_orig_frame = cv2.resize(orig_frame, (1280, 720))

        video_writer.write(resized_orig_frame)
        pbar_video.update(1)
        # 每%10更新一次进度条
        if frame_count % 10 == 0:  # 每10帧更新一次，更频繁
            update_progress(action_id, frame_count, total_frames, phase=1)

    if M is not None and data and out_warped_video_file:
        warped_frame = cv2.warpPerspective(last_orig_frame, M, (1280, 720))
        for keypoints in data[-1:]:  # 只画最后一帧
            for i, (x, y) in enumerate(keypoints):
                if i == 2:
                    color = (255, 0, 0)  # 左脚踝蓝色
                elif i == 3:
                    color = (0, 0, 255)  # 右脚踝红色
                cv2.circle(warped_frame, (int(x), int(y)), 2, color, 2)
            #cv2.line(warped_frame, (int(keypoints[2, 0]), int(keypoints[2, 1])), 
                     #(int(keypoints[3, 0]), int(keypoints[3, 1])), (255, 0, 0), 1)
        resized_warped_frame = cv2.resize(warped_frame, (1280, 720))

        warped_video_writer.write(resized_warped_frame)
        
    cap.release()
    video_writer.release()
    if warped_video_writer:
        warped_video_writer.release()
    cv2.destroyAllWindows()
    
    if M is None or not key_points:
        print("No perspective matrix calculated, check video or models!")
    else:
        # 过滤掉左右脚踝坐标中任一值为0的时间点数据
        filtered_data = []
        for idx, keypoints in enumerate(data):
            left_ankle = keypoints[6]
            right_ankle = keypoints[7]
            if left_ankle[0] != 0 and left_ankle[1] != 0 and right_ankle[0] != 0 and right_ankle[1] != 0:
                filtered_data.append(keypoints)
    
        # 更新 data 列表为过滤后的数据
        data = filtered_data
        
        # 保存左右脚脚踝关键点坐标（只保存左右脚踝）
    
        print(f"Video processing completed. Total frames: {frame_count}")

        
         # 渲染参数到视频
        def render_parameters_to_video(input_video_path, output_video_path, scaled_data, smoothed_speed):
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
            
            '''
            # 读取步态参数
            with open('data/gait_parameters.json', 'r') as f:
                gait_params = json.load(f)
            avg_step_length = gait_params['avg_step_length']
            avg_step_speed = gait_params['avg_step_speed']
            '''

            # 打开输入视频
            cap = cv2.VideoCapture(input_video_path)

            total_render_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            pbar_render = tqdm(total=total_render_frames, desc="Rendering parameters")  # 新增进度条

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 创建输出视频
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                current_time = frame_count / fps
                frame_count += 1

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
                text_x = width - 400
                text_y = 50

                # 绘制背景矩形
                cv2.rectangle(frame, (text_x - 10, text_y - text_height - 5), 
                            (text_x + text_width + 10, text_y + 5), 
                            background_color, -1)

                # 绘制文字
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness)

                # 第二行文字
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

                # 第三行文字
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

                # 第四行文字
                text_y += 40
                text = f"Right Speed: {right_speed:.2f}m/s"
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_width, text_height = text_size

                # 绘制背景矩形
                cv2.rectangle(frame, (text_x - 10, text_y - text_height - 5), 
                            (text_x + text_width + 10, text_y + 5), 
                            background_color, -1)

                # 绘制文字
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness)
                
                # 第五行文字
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
                
                '''
                text_y += 40
                cv2.putText(frame, f"Avg Step Length: {avg_step_length:.2f}m", 
                            (width - 400, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 255, 0), 2)
                text_y += 40
                cv2.putText(frame, f"Avg Step Speed: {avg_step_speed:.2f}m/s", 
                            (width - 400, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 255, 0), 2)
                '''
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
        scaled_data, times, left_speed, right_speed, smoothed_data, result = caculate_output(key_points, np.array(data), out_json_file, M)
        # 准备速度数据
        smoothed_speed = [{'time': t, 'right_speed': rs} for t, rs in zip(times, right_speed)]
        # 渲染参数到原始视频
        render_parameters_to_video(mid_video_file, out_video_file, scaled_data, smoothed_speed)
        os.system(
            f"ffmpeg -i {out_video_file} -c:v libx264 -c:a aac {out_video_file.replace('.mp4', '_final.mp4')}")
        os.system(
            f"mv -f {out_video_file.replace('.mp4', '_final.mp4')} {out_video_file}")
        return result

if __name__ == "__main__":
    if len(sys.argv) not in [4, 5]:
        print("Usage: python inference.py input_video.mp4 out_video.mp4 out_json.json [out_warped_video.mp4]")
        sys.exit(-1)
    out_warped_video_file = sys.argv[4] if len(sys.argv) > 4 else None
    main(sys.argv[1], sys.argv[2], sys.argv[3], out_warped_video_file)
