import cv2
import numpy as np
import json
import csv
import matplotlib.pyplot as plt
from ultralytics import YOLO
from scipy.signal import savgol_filter, find_peaks
import copy
import gait_analysis_v9 as gait_analysis # 导入步态分析模块

def default_serializer(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def diff_non_zero(arr):
    non_zero_values = [i for i in arr if i != 0]
    return max(non_zero_values) - min(non_zero_values) if non_zero_values else 0

def sort_points(points):
    points_sorted_by_x = sorted(points, key=lambda p: p[0])
    x_first_two = points_sorted_by_x[:2]
    x_last_two = points_sorted_by_x[2:]
    x_first_two_sorted = sorted(x_first_two, key=lambda p: p[1])
    x_last_two_sorted = sorted(x_last_two, key=lambda p: p[1])
    return [x_first_two_sorted[0], x_last_two_sorted[0], x_first_two_sorted[1], x_last_two_sorted[1]]

def get_featurepoints2(image0):
    return False, []

def smooth_data(data, window_size):
    smoothed_data = []
    n = len(data)
    for i in range(n):
        start = max(0, i - window_size // 2)
        end = min(n, i + window_size // 2 + 1)
        window = data[start:end]
        smoothed_time = data[i]['time']
        smoothed_left = [sum(d['left_ankle'][0] for d in window) / len(window),
                         sum(d['left_ankle'][1] for d in window) / len(window)]
        smoothed_right = [sum(d['right_ankle'][0] for d in window) / len(window),
                          sum(d['right_ankle'][1] for d in window) / len(window)]
        smoothed_left_shoulder = [sum(d['left_shoulder'][0] for d in window) / len(window),
                                 sum(d['left_shoulder'][1] for d in window) / len(window)]
        smoothed_right_shoulder = [sum(d['right_shoulder'][0] for d in window) / len(window),
                                   sum(d['right_shoulder'][1] for d in window) / len(window)]
        smoothed_left_hip = [sum(d['left_hip'][0] for d in window) / len(window),
                             sum(d['left_hip'][1] for d in window) / len(window)]
        smoothed_right_hip = [sum(d['right_hip'][0] for d in window) / len(window),
                              sum(d['right_hip'][1] for d in window) / len(window)]
        smoothed_left_knee = [sum(d['left_knee'][0] for d in window) / len(window),
                             sum(d['left_knee'][1] for d in window) / len(window)]
        smoothed_right_knee = [sum(d['right_knee'][0] for d in window) / len(window),
                              sum(d['right_knee'][1] for d in window) / len(window)]
        smoothed_data.append({
            'time': smoothed_time,
            'left_ankle': smoothed_left,
            'right_ankle': smoothed_right,
            'left_shoulder': smoothed_left_shoulder,
            'right_shoulder': smoothed_right_shoulder,
            'left_hip': smoothed_left_hip,
            'right_hip': smoothed_right_hip,
            'left_knee': smoothed_left_knee,
            'right_knee': smoothed_right_knee
        })
    return smoothed_data

def calculate_speed(data, fps=24.0):
    data.sort(key=lambda x: x['time'])
    window_size = 7
    smoothed_data = smooth_data(data, window_size)
    
    # 计算速度
    n = len(smoothed_data)
    times = [d['time'] for d in smoothed_data]
    left_speed = [0.0] * n
    right_speed = [0.0] * n

    for i in range(1, n):
        prev = smoothed_data[i-1]
        current = smoothed_data[i]
        delta_t = current['time'] - prev['time']
        if delta_t > 1/12 + 1e-9:
            continue
        cl = current['left_ankle']
        cr = current['right_ankle']
        pl = prev['left_ankle']
        pr = prev['right_ankle']
        current_valid = (cl[0] != 0 or cl[1] != 0) and (cr[0] != 0 or cr[1] != 0)
        prev_valid = (pl[0] != 0 or pl[1] != 0) and (pr[0] != 0 or pr[1] != 0)
        if current_valid and prev_valid:
            dx_left = cl[0] - pl[0]
            dx_right = cr[0] - pr[0]
            left_speed[i] = dx_left / delta_t
            right_speed[i] = dx_right / delta_t

    # 使用Savitzky-Golay滤波器平滑速度曲线
    left_speed = savgol_filter(left_speed, window_length=5, polyorder=2)
    right_speed = savgol_filter(right_speed, window_length=5, polyorder=2)

    return times, left_speed, right_speed, smoothed_data

def find_peaks_and_gait_points(times, left_speed, right_speed, smoothed_data, out_file, y_scale, z_x_scale):
    # 找到波峰
    left_peaks_indices, _ = find_peaks(left_speed, prominence=1, distance=12)
    right_peaks_indices, _ = find_peaks(right_speed, prominence=1, distance=12)
    left_peaks = [(times[i], left_speed[i]) for i in left_peaks_indices]
    right_peaks = [(times[i], right_speed[i]) for i in right_peaks_indices]
    
    filtered_left_peaks = left_peaks
    filtered_right_peaks = right_peaks
    
    # 找到速度的局部最小值（波谷）
    left_min_indices, _ = find_peaks(-np.array(left_speed))
    right_min_indices, _ = find_peaks(-np.array(right_speed))

    # 去除速度值大于0.5的波谷数据
    left_min_indices = [i for i in left_min_indices if left_speed[i] <= 0.5]
    right_min_indices = [i for i in right_min_indices if right_speed[i] <= 0.5]

    left_points = []
    for i in range(len(filtered_left_peaks)):
        l_peak_index = left_peaks_indices[i]  # 当前波峰在原始数据中的索引
        
        # 找到左侧最近的波谷
        l_lift_mins = [m for m in left_min_indices if m < l_peak_index]
        l_lift_index = l_lift_mins[-1] if l_lift_mins else 0
        
        # 找到右侧最近的波谷
        l_land_mins = [m for m in left_min_indices if m > l_peak_index]
        l_land_index = l_land_mins[0] if l_land_mins else len(left_speed)-1
        
        l_lift_time = times[l_lift_index]
        l_land_time = times[l_land_index]
        
        l_lift_idx = np.argmin(np.abs(np.array(times) - l_lift_time))
        l_land_idx = np.argmin(np.abs(np.array(times) - l_land_time))

        l_lift_ponit_x = smoothed_data[l_lift_idx]['left_ankle'][0]
        l_lift_ponit_y = smoothed_data[l_lift_idx]['left_ankle'][1]
        l_land_ponit_x = smoothed_data[l_land_idx]['left_ankle'][0]
        l_land_ponit_y = smoothed_data[l_land_idx]['left_ankle'][1]

        l_lift_ponit_frame = int(l_lift_time * 24)
        l_land_ponit_frame = int(l_land_time * 24)

        left_points.append((l_lift_time, l_land_time, l_lift_ponit_x, l_lift_ponit_y, l_land_ponit_x, l_land_ponit_y, l_lift_ponit_frame, l_land_ponit_frame))
    
    right_points = []
    for i in range(len(filtered_right_peaks)):
        r_peak_index = right_peaks_indices[i]  # 当前波峰在原始数据中的索引
        
        # 找到左侧最近的波谷
        r_lift_mins = [m for m in right_min_indices if m < r_peak_index]
        r_lift_index = r_lift_mins[-1] if r_lift_mins else 0
        
        # 找到右侧最近的波谷
        r_land_mins = [m for m in right_min_indices if m > r_peak_index]
        r_land_index = r_land_mins[0] if r_land_mins else len(right_speed)-1
        
        r_lift_time = times[r_lift_index]
        r_land_time = times[r_land_index]
        
        r_lift_idx = np.argmin(np.abs(np.array(times) - r_lift_time))
        r_land_idx = np.argmin(np.abs(np.array(times) - r_land_time))

        r_lift_ponit_x = smoothed_data[r_lift_idx]['right_ankle'][0]
        r_lift_ponit_y = smoothed_data[r_lift_idx]['right_ankle'][1]
        r_land_ponit_x = smoothed_data[r_land_idx]['right_ankle'][0]
        r_land_ponit_y = smoothed_data[r_land_idx]['right_ankle'][1]

        r_lift_ponit_frame = int(r_lift_time * 24)
        r_land_ponit_frame = int(r_land_time * 24)

        right_points.append((r_lift_time, r_land_time, r_lift_ponit_x, r_lift_ponit_y, r_land_ponit_x, r_land_ponit_y, r_lift_ponit_frame, r_land_ponit_frame))

    # 调用步态分析模块
    result = gait_analysis.calculate_gait_parameters(left_points, right_points, smoothed_data, y_scale, z_x_scale)
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=4, default=default_serializer)
    
    return left_peaks, right_peaks, left_points, right_points, filtered_left_peaks, filtered_right_peaks, result


def caculate_output(key_points4, datas, out_file, M, fps=24.0):
    box_len = 3.0
    box_width = 0.5

    # 计算透视变换后的坐标范围
    transformed_points = cv2.perspectiveTransform(np.array([key_points4], dtype='float32'), M)[0]
    min_x = min(p[0] for p in transformed_points)
    max_x = max(p[0] for p in transformed_points)
    min_y = min(p[1] for p in transformed_points)
    max_y = max(p[1] for p in transformed_points)

    # 计算缩放因子
    x_scale = box_len / (max_x - min_x)
    y_scale = box_width / (max_y - min_y)

    # 用于计算足离地高度的缩放因子，由于x轴与z轴垂直，认为缩放因子一致，故使用x轴的缩放因子
    z_x_scale = box_len / ((key_points4[3][0]+key_points4[2][0])/2 - (key_points4[0][0]+key_points4[1][0])/2)

    data_list = []
    times = []
    left_x = []
    left_y = []
    right_x = []
    right_y = []
    for frame_idx, frame_data in enumerate(datas):
        time = frame_idx / fps
        times.append(time)
        left_shoulder = frame_data[0]
        right_shoulder = frame_data[1]
        left_hip = frame_data[2]
        right_hip = frame_data[3]
        left_knee = frame_data[4]
        right_knee = frame_data[5]
        left_ankle = frame_data[6]
        right_ankle = frame_data[7]
        left_ankle_transformed = cv2.perspectiveTransform(np.array([[left_ankle]], dtype='float32'), M)[0][0]
        right_ankle_transformed = cv2.perspectiveTransform(np.array([[right_ankle]], dtype='float32'), M)[0][0]
        data_list.append({
            'time': time,
            'left_shoulder': [left_shoulder[0], left_shoulder[1]],
            'right_shoulder': [right_shoulder[0], right_shoulder[1]],
            'left_hip': [left_hip[0], left_hip[1]],
            'right_hip': [right_hip[0], right_hip[1]],
            'left_knee': [left_knee[0], left_knee[1]],
            'right_knee': [right_knee[0], right_knee[1]],
            'left_ankle': [left_ankle_transformed[0], left_ankle_transformed[1]],
            'right_ankle': [right_ankle_transformed[0], right_ankle_transformed[1]]
        })
        left_x.append(left_ankle_transformed[0])
        left_y.append(left_ankle_transformed[1])
        right_x.append(right_ankle_transformed[0])
        right_y.append(right_ankle_transformed[1])
    
    # 单位转换
    scaled_data = []
    for d in data_list:
        scaled_data.append({
            'time': d['time'],
            'left_ankle': [
                d['left_ankle'][0] * x_scale,
                d['left_ankle'][1] * y_scale
            ],
            'right_ankle': [
                d['right_ankle'][0] * x_scale,
                d['right_ankle'][1] * y_scale
            ],
            'left_shoulder': [
                d['left_shoulder'][0] * z_x_scale,
                d['left_shoulder'][1] * z_x_scale
            ],
            'right_shoulder': [
                d['right_shoulder'][0] * z_x_scale,
                d['right_shoulder'][1] * z_x_scale
            ],
            'left_hip': [
                d['left_hip'][0] * z_x_scale,
                d['left_hip'][1] * z_x_scale
            ],
            'right_hip': [
                d['right_hip'][0] * z_x_scale,
                d['right_hip'][1] * z_x_scale
            ],
            'left_knee': [
                d['left_knee'][0] * z_x_scale,
                d['left_knee'][1] * z_x_scale
            ], 
            'right_knee': [
                d['right_knee'][0] * z_x_scale,
                d['right_knee'][1] * z_x_scale
            ]
        })

    times, left_speed, right_speed, smoothed_data = calculate_speed(scaled_data)
    left_peaks, right_peaks, left_midpoints, right_midpoints, filtered_left_peaks, filtered_right_peaks, result = find_peaks_and_gait_points(times, left_speed, right_speed, smoothed_data, out_file, y_scale, z_x_scale)
    return scaled_data, times, left_speed, right_speed, smoothed_data, result


if __name__ == "__main__":
    pass