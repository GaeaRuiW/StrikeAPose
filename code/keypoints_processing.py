import cv2
import numpy as np
import json
import csv
import matplotlib.pyplot as plt
from ultralytics import YOLO
from scipy.signal import savgol_filter, find_peaks
import copy
import gait_analysis as gait_analysis # 导入步态分析模块

# 定义一个默认的序列化函数
def default_serializer(obj):
    # 如果对象是np.float32类型
    if isinstance(obj, np.float32):
        # 返回float类型
        return float(obj)
    # 否则抛出异常，提示对象不可序列化
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

# 用于对一组数据进行平滑处理
def smooth_data(data, window_size):
    # 定义一个空列表，用于存储平滑后的数据
    smoothed_data = []
    # 获取数据的长度
    n = len(data)
    # 遍历数据
    for i in range(n):
        # 计算窗口的起始和结束位置
        start = max(0, i - window_size // 2)
        end = min(n, i + window_size // 2 + 1)
        # 获取窗口内的数据
        window = data[start:end]
        # 获取当前时间
        smoothed_time = data[i]['time']
        # 计算左脚踝的平滑值
        smoothed_left = [sum(d['left_ankle'][0] for d in window) / len(window),
                         sum(d['left_ankle'][1] for d in window) / len(window)]
        # 计算右脚踝的平滑值
        smoothed_right = [sum(d['right_ankle'][0] for d in window) / len(window),
                          sum(d['right_ankle'][1] for d in window) / len(window)]
        # 计算左肩膀的平滑值
        smoothed_left_shoulder = [sum(d['left_shoulder'][0] for d in window) / len(window),
                                 sum(d['left_shoulder'][1] for d in window) / len(window)]
        # 计算右肩膀的平滑值
        smoothed_right_shoulder = [sum(d['right_shoulder'][0] for d in window) / len(window),
                                   sum(d['right_shoulder'][1] for d in window) / len(window)]
        # 计算左臀的平滑值
        smoothed_left_hip = [sum(d['left_hip'][0] for d in window) / len(window),
                             sum(d['left_hip'][1] for d in window) / len(window)]
        # 计算右臀的平滑值
        smoothed_right_hip = [sum(d['right_hip'][0] for d in window) / len(window),
                              sum(d['right_hip'][1] for d in window) / len(window)]
        # 计算左膝的平滑值
        smoothed_left_knee = [sum(d['left_knee'][0] for d in window) / len(window),
                             sum(d['left_knee'][1] for d in window) / len(window)]
        # 计算右膝的平滑值
        smoothed_right_knee = [sum(d['right_knee'][0] for d in window) / len(window),
                              sum(d['right_knee'][1] for d in window) / len(window)]
        smoothed_left_ankle_z = [sum(d['left_ankle_z'][0] for d in window) / len(window),
                                 sum(d['left_ankle_z'][1] for d in window) / len(window)]
        smoothed_right_ankle_z = [sum(d['right_ankle_z'][0] for d in window) / len(window),
                                  sum(d['right_ankle_z'][1] for d in window) / len(window)]
        # 将平滑后的数据添加到列表中
        smoothed_data.append({
            'time': smoothed_time,
            'left_ankle': smoothed_left,
            'right_ankle': smoothed_right,
            'left_shoulder': smoothed_left_shoulder,
            'right_shoulder': smoothed_right_shoulder,
            'left_hip': smoothed_left_hip,
            'right_hip': smoothed_right_hip,
            'left_knee': smoothed_left_knee,
            'right_knee': smoothed_right_knee,
            'left_ankle_z': smoothed_left_ankle_z,
            'right_ankle_z': smoothed_right_ankle_z
        })
    
    '''
    # 绘制右脚曲线
    times = [d['time'] for d in smoothed_data]
    right_ankle_x = [d['right_ankle'][0] for d in smoothed_data]
    plt.figure(figsize=(12, 6))
    plt.plot(times, right_ankle_x, label='Right Coordinates', color='red')

    plt.title('Right Ankle X Coordinates')
    plt.xlabel('Time (s)')
    plt.ylabel('X Coordinate')
    plt.legend()
    plt.grid(True)
    plt.savefig('data/right Ankle X Coordinates.png')
    plt.close()

    # 绘制左脚曲线
    left_ankle_x = [d['left_ankle'][0] for d in smoothed_data]
    plt.figure(figsize=(12, 6))
    plt.plot(times, left_ankle_x, label='Left Coordinates', color='red')

    plt.title('Left Ankle X Coordinates')
    plt.xlabel('Time (s)')
    plt.ylabel('X Coordinate')
    plt.legend()
    plt.grid(True)
    plt.savefig('data/Left Ankle X Coordinates.png')
    plt.close()
    '''
    # 返回平滑后的数据
    return smoothed_data

# 用于计算速度
def calculate_speed(data, fps=30.0):
    # 对数据进行排序
    data.sort(key=lambda x: x['time'])
    # 设置窗口大小
    window_size = 17
    # 对数据进行平滑处理
    smoothed_data = smooth_data(data, window_size)
    
    # 计算速度
    n = len(smoothed_data)
    # 获取时间序列
    times = [d['time'] for d in smoothed_data]
    # 初始化左右脚速度
    left_speed = [0.0] * n
    right_speed = [0.0] * n

    # 遍历平滑后的数据
    for i in range(1, n):
        # 获取前后两个数据点
        prev = smoothed_data[i-1]
        current = smoothed_data[i]
        # 计算时间差
        delta_t = current['time'] - prev['time']
        # 如果时间差大于1/12秒，则跳过
        if delta_t > 1/12 + 1e-9:
            continue
        # 获取左右脚的坐标
        cl = current['left_ankle']
        cr = current['right_ankle']
        pl = prev['left_ankle']
        pr = prev['right_ankle']
        # 判断当前数据点是否有效
        current_valid = (cl[0] != 0 or cl[1] != 0) and (cr[0] != 0 or cr[1] != 0)
        prev_valid = (pl[0] != 0 or pl[1] != 0) and (pr[0] != 0 or pr[1] != 0)
        # 如果当前数据点和前一个数据点都有效，则计算速度
        if current_valid and prev_valid:
            # 计算左右脚的速度
            dx_left = cl[0] - pl[0]
            dx_right = cr[0] - pr[0]
            left_speed[i] = dx_left / delta_t
            right_speed[i] = dx_right / delta_t

    # 使用Savitzky-Golay滤波器平滑速度曲线
    left_speed = savgol_filter(left_speed, window_length=17, polyorder=2)
    right_speed = savgol_filter(right_speed, window_length=17, polyorder=2)
    '''
    # 绘制右脚速度曲线
    plt.figure(figsize=(12, 6))
    plt.plot(times, right_speed, label='Right Speed', color='red')

    plt.title('Right Ankle Speed')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed')
    plt.legend()
    plt.grid(True)
    plt.savefig('data/right Speed.png')
    plt.close()

    # 绘制左脚速度曲线
    plt.figure(figsize=(12, 6))
    plt.plot(times, left_speed, label='Left Speed', color='red')

    plt.title('Left Ankle Speed')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed')
    plt.legend()
    plt.grid(True)
    plt.savefig('data/Left Speed.png')
    plt.close()
    '''    

    # 返回时间序列、左右脚速度和平滑后的数据
    return times, left_speed, right_speed, smoothed_data

# 用于找到波峰和步态点
def find_peaks_and_gait_points(times, left_speed, right_speed, smoothed_data, out_file, y_scale, z_x_scale):
    prominence = 0.3
    height = 0.1
    fps=30
    # 找到波峰（正向波峰）
    left_peaks_indices, _ = find_peaks(left_speed, prominence=prominence, distance=12, height=height)
    right_peaks_indices, _ = find_peaks(right_speed, prominence=prominence, distance=12, height=height)
    
    left_turn = None
    right_turn = None

    n_left_peaks_indices, _ = find_peaks(-left_speed, prominence=prominence, distance=12, height=height)
    n_right_peaks_indices, _ = find_peaks(-right_speed, prominence=prominence, distance=12, height=height)

    if (np.any(left_speed < -0.4) or np.any(right_speed < -0.4)) and (len(n_left_peaks_indices) > 0 and len(n_right_peaks_indices) > 0) and (len(left_peaks_indices) > 0 and len(right_peaks_indices) > 0):
        # 速度取负，找到波峰（返程波峰）
        n_left_peaks_indices, _ = find_peaks(-left_speed, prominence=prominence, distance=12, height=height)
        n_right_peaks_indices, _ = find_peaks(-right_speed, prominence=prominence, distance=12, height=height)

        left_turn = int(((left_peaks_indices[-1] + n_left_peaks_indices[0]) / 2))
        left_p = left_speed[:left_turn]
        left_n = -left_speed[left_turn:]
        left_speed = np.concatenate((left_p, left_n))
        
        right_turn = int(((right_peaks_indices[-1] + n_right_peaks_indices[0]) / 2))
        right_p = right_speed[:right_turn]
        right_n = -right_speed[right_turn:]
        right_speed = np.concatenate((right_p, right_n))

    # 找到波峰
    left_peaks_indices, _ = find_peaks(left_speed, prominence=prominence, distance=12, height=height)
    right_peaks_indices, _ = find_peaks(right_speed, prominence=prominence, distance=12, height=height)

    left_peaks = [(times[i], left_speed[i]) for i in left_peaks_indices]
    right_peaks = [(times[i], right_speed[i]) for i in right_peaks_indices]
    
    filtered_left_peaks = left_peaks
    filtered_right_peaks = right_peaks
    
    # 找到速度的局部最小值（波谷）
    left_min_indices, _ = find_peaks(-np.array(left_speed))
    right_min_indices, _ = find_peaks(-np.array(right_speed))

    # 去除速度值大于0.8的波谷数据
    left_min_indices = [i for i in left_min_indices if left_speed[i] <= 0.8]
    right_min_indices = [i for i in right_min_indices if right_speed[i] <= 0.8]

    left_points = []
    for i in range(len(filtered_left_peaks)):
        l_peak_index = left_peaks_indices[i]  # 当前波峰在原始数据中的索引
        
        # 找到左侧最近的波谷
        l_lift_mins = [m for m in left_min_indices if m < l_peak_index]
        l_lift_index = l_lift_mins[-1] if l_lift_mins else 0
        
        # 找到右侧最近的波谷
        l_land_mins = [m for m in left_min_indices if m > l_peak_index]
        #l_land_index = l_land_mins[0] if l_land_mins else len(left_speed)-1
        # 如果当前是最后一个波峰且右侧无波谷，取最后一个点
        if i == len(filtered_left_peaks) - 1 and not l_land_mins:
            l_land_index = len(left_speed) - 1
        else:
            l_land_index = l_land_mins[0] if l_land_mins else len(left_speed)-1

        l_lift_time = times[l_lift_index]
        l_land_time = times[l_land_index]
        
        l_lift_idx = np.argmin(np.abs(np.array(times) - l_lift_time))
        l_land_idx = np.argmin(np.abs(np.array(times) - l_land_time))

        l_lift_ponit_x = smoothed_data[l_lift_idx]['left_ankle'][0]
        l_lift_ponit_y = smoothed_data[l_lift_idx]['left_ankle'][1]
        l_land_ponit_x = smoothed_data[l_land_idx]['left_ankle'][0]
        l_land_ponit_y = smoothed_data[l_land_idx]['left_ankle'][1]

        l_lift_ponit_frame = int(l_lift_time * fps)
        l_land_ponit_frame = int(l_land_time * fps)

        left_points.append((l_lift_time, l_land_time, l_lift_ponit_x, l_lift_ponit_y, l_land_ponit_x, l_land_ponit_y, l_lift_ponit_frame, l_land_ponit_frame))
    
    right_points = []
    for i in range(len(filtered_right_peaks)):
        r_peak_index = right_peaks_indices[i]  # 当前波峰在原始数据中的索引
        
        # 找到左侧最近的波谷
        r_lift_mins = [m for m in right_min_indices if m < r_peak_index]
        r_lift_index = r_lift_mins[-1] if r_lift_mins else 0
        
        # 找到右侧最近的波谷
        r_land_mins = [m for m in right_min_indices if m > r_peak_index]
        #r_land_index = r_land_mins[0] if r_land_mins else len(right_speed)-1
        # 如果当前是最后一个波峰且右侧无波谷，取最后一个点
        if i == len(filtered_right_peaks) - 1 and not r_land_mins:
            r_land_index = len(right_speed) - 1
        else:
            r_land_index = r_land_mins[0] if r_land_mins else len(right_speed)-1

        r_lift_time = times[r_lift_index]
        r_land_time = times[r_land_index]
        
        r_lift_idx = np.argmin(np.abs(np.array(times) - r_lift_time))
        r_land_idx = np.argmin(np.abs(np.array(times) - r_land_time))

        r_lift_ponit_x = smoothed_data[r_lift_idx]['right_ankle'][0]
        r_lift_ponit_y = smoothed_data[r_lift_idx]['right_ankle'][1]
        r_land_ponit_x = smoothed_data[r_land_idx]['right_ankle'][0]
        r_land_ponit_y = smoothed_data[r_land_idx]['right_ankle'][1]

        r_lift_ponit_frame = int(r_lift_time * fps)
        r_land_ponit_frame = int(r_land_time * fps)

        right_points.append((r_lift_time, r_land_time, r_lift_ponit_x, r_lift_ponit_y, r_land_ponit_x, r_land_ponit_y, r_lift_ponit_frame, r_land_ponit_frame))

    '''
    # 绘制右脚速度曲线、波峰和最近波谷
    plt.figure(figsize=(12, 6))
    plt.plot(times, right_speed, label='Right Speed', color='red')
    plt.scatter([times[i] for i in right_peaks_indices], [right_speed[i] for i in right_peaks_indices], 
                color='red', label='Peaks')
    plt.scatter([times[i] for i in right_min_indices], [right_speed[i] for i in right_min_indices], 
                color='blue', label='Valleys')

      
    plt.title('Right Foot Speed with Peaks and Nearest Valleys')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed')
    plt.legend()
    plt.grid(True)
    plt.savefig('data/right_speed_with_peaks_and_valleys_.png')
    plt.close()

    # 绘制左脚速度曲线、波峰和最近波谷
    plt.figure(figsize=(12, 6))
    plt.plot(times, left_speed, label='left Speed', color='red')
    plt.scatter([times[i] for i in left_peaks_indices], [left_speed[i] for i in left_peaks_indices], 
                color='red', label='Peaks')
    plt.scatter([times[i] for i in left_min_indices], [left_speed[i] for i in left_min_indices], 
                color='blue', label='Valleys')

    plt.title('left Foot Speed with Peaks and Nearest Valleys')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed')
    plt.legend()
    plt.grid(True)
    plt.savefig('data/left_speed_with_peaks_and_valleys_.png')
    plt.close()
    '''
    if left_turn is None or right_turn is None:
        return left_peaks, right_peaks, left_points, right_points, filtered_left_peaks, filtered_right_peaks, None
    # 调用步态分析模块
    result = gait_analysis.calculate_gait_parameters(left_points, right_points, smoothed_data, left_turn, right_turn)
    # with open(out_file, 'w') as f:
    #     json.dump(result, f, indent=4, default=default_serializer)
    
    return left_peaks, right_peaks, left_points, right_points, filtered_left_peaks, filtered_right_peaks, result

# 用于计算输出
def caculate_output(key_points4, datas, out_file, M, fps=30.0):
    # 定义标定物的长度和宽度
    box_len = 3
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
    # 遍历每一帧数据
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
        # 计算足部坐标的透视变换
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
            'right_ankle': [right_ankle_transformed[0], right_ankle_transformed[1]],
            'left_ankle_z': [left_ankle[0], left_ankle[1]],
            'right_ankle_z': [right_ankle[0], right_ankle[1]]
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
            ],
            'left_ankle_z': [
                d['left_ankle_z'][0] * z_x_scale,
                d['left_ankle_z'][1] * z_x_scale
            ],
            'right_ankle_z': [
                d['right_ankle_z'][0] * z_x_scale,
                d['right_ankle_z'][1] * z_x_scale
            ]
        })
    # 如果长度小于17直接return
    if len(scaled_data) < 17:
        return scaled_data, [], [], [], [], None
    # 计算速度
    times, left_speed, right_speed, smoothed_data = calculate_speed(scaled_data)
    # 调用步态分析模块
    left_peaks, right_peaks, left_midpoints, right_midpoints, filtered_left_peaks, filtered_right_peaks, result = find_peaks_and_gait_points(times, left_speed, right_speed, smoothed_data, out_file, y_scale, z_x_scale)
    return scaled_data, times, left_speed, right_speed, smoothed_data, result


if __name__ == "__main__":
    pass
