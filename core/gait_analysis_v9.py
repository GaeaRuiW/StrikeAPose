# gait_analysis_v9.py

import numpy as np
from collections import defaultdict

def calculate_angle(a, b, c):
    """计算三点之间的夹角（单位：弧度）"""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    dot_product = np.dot(ba, bc)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cosine_angle = dot_product / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    # 计算弧度后转换为角度
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def calculate_gait_parameters(left_points, right_points, smoothed_data, y_scale, z_x_scale, fps=24.0):
    # 构造步骤列表
    left_steps = []
    for lp in left_points:
        lift_time, land_time, lift_x, lift_y, land_x, land_y, lift_frame, land_frame = lp
        left_steps.append({
            'start_time': lift_time,
            'end_time': land_time,
            'leg': 'left',
            'lift_x': lift_x,
            'lift_y': lift_y,
            'land_x': land_x,
            'land_y': land_y,
            'start_frame': lift_frame,
            'end_frame': land_frame
        })
    
    right_steps = []
    for rp in right_points:
        lift_time, land_time, lift_x, lift_y, land_x, land_y, lift_frame, land_frame = rp
        right_steps.append({
            'start_time': lift_time,
            'end_time': land_time,
            'leg': 'right',
            'lift_x': lift_x,
            'lift_y': lift_y,
            'land_x': land_x,
            'land_y': land_y,
            'start_frame': lift_frame,
            'end_frame': land_frame
        })
    
    all_steps = sorted(left_steps + right_steps, key=lambda x: x['start_time'])
    
    # 计算步宽
    left_land_ys = [step['land_y'] for step in all_steps if step['leg'] == 'left']
    right_land_ys = [step['land_y'] for step in all_steps if step['leg'] == 'right']
    avg_left_y = np.mean(left_land_ys) if left_land_ys else 0
    avg_right_y = np.mean(right_land_ys) if right_land_ys else 0
    step_width_value = abs(avg_left_y - avg_right_y)
    
    # 获取首尾关键点
    first_step = all_steps[1]
    last_step = all_steps[-2]
    
    # 计算总位移（注意取绝对值）
    start_x = first_step['lift_x']  # 第一个离地点x坐标
    end_x = last_step['land_x']     # 最后一个着地点x坐标
    total_distance = abs(end_x - start_x)
    
    # 计算总时间（注意使用第一个开始时间和最后一个结束时间）
    total_time = last_step['end_time'] - first_step['start_time']
    
    # 计算整体速度
    average_speed = total_distance / total_time if total_time > 0 else 0

    # 初始化stride_data和跟踪另一只脚的上一次land_x
    last_left_stride_x = None  # 用于同侧步幅计算
    last_right_stride_x = None  # 用于同侧步幅计算
    last_left_land_x = None   # 用于对侧步长计算
    last_right_land_x = None  # 用于对侧步长计算
    steps_info = []
    prev_step_info = None  # 跟踪前一步的step_info

    for i, step in enumerate(all_steps):
        leg = step['leg']
        start_time = step['start_time']
        end_time = step['end_time']
        start_frame = step['start_frame']
        end_frame = step['end_frame']
        lift_x = step['lift_x']
        land_x = step['land_x']
        land_y = step['land_y']
        
        if i == 0 or i == len(all_steps) - 1:
            # 跳过无效的第一步
            if leg == 'left':
                last_left_land_x = land_x
                
            else:
                last_right_land_x = land_x
            continue
        
        # 计算步长
        if leg == 'left':
            if last_right_land_x is not None:
                step_length = abs(land_x - last_right_land_x)
            else:
                step_length = 0
            last_left_land_x = land_x  # 更新左脚的着地点
        else:
            if last_left_land_x is not None:
                step_length = abs(land_x - last_left_land_x)
            else:
                step_length = 0
            last_right_land_x = land_x  # 更新右脚的着地点
        
        # 计算步幅
        stride_length = abs(land_x - lift_x)

        # 计算步速
        time_diff = end_time - start_time
        # step_speed = step_length / time_diff if time_diff > 0 else 0
        
        # 计算支撑时间
        support_time = time_diff
        
        # 计算前脚
        front_leg = leg
        
        # 计算离地高度
        frames_in_step = [d for d in smoothed_data if start_time <= d['time'] <= end_time]
        if leg == 'left':
            ankle_ys = [d['left_ankle'][1] for d in frames_in_step]
        else:
            ankle_ys = [d['right_ankle'][1] for d in frames_in_step]
        if ankle_ys:
            max_y = max(ankle_ys)
            min_y = min(ankle_ys)
            liftoff_height = (max_y - min_y) / y_scale * z_x_scale - 0.12
        else:
            liftoff_height = 0
        
        # 计算髋关节角度
        angles = []
        for d in frames_in_step:
            if leg == 'left':
                a = (d['left_shoulder'][0], d['left_shoulder'][1])
                b = (d['left_hip'][0], d['left_hip'][1])
                c = (d['left_knee'][0], d['left_knee'][1])
            else:
                a = (d['right_shoulder'][0], d['right_shoulder'][1])
                b = (d['right_hip'][0], d['right_hip'][1])
                c = (d['right_knee'][0], d['right_knee'][1])
            angle = calculate_angle(a, b, c)
            angles.append(angle)
        hip_min = np.min(angles) if angles else 0
        hip_max = np.max(angles) if angles else 0
        
        # 判断第一步
        first_step = (i == 1)

        current_step_length = step_length

        # 构建step_info
        step_info = {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'step_length': step_length,
            'step_width': step_width_value,
            'step_speed': average_speed,
            'front_leg': front_leg,
            'support_time': support_time,
            'liftoff_height': liftoff_height,
            'hip_min_degree': hip_min,
            'hip_max_degree': hip_max,
            'first_step': first_step,
            'steps_diff': 0,
            'stride_length': stride_length
        }

        # 如果存在前一步，计算并更新前一步的步长差
        if prev_step_info is not None:
            steps_diff = current_step_length - prev_step_info['step_length']
            prev_step_info['steps_diff'] = steps_diff

        steps_info.append(step_info)
        prev_step_info = step_info  # 更新前一步为当前步骤
    
    # 构建返回结果
    return [{
        "stage_n": 1,
        "start_frame": steps_info[0]['start_frame'] if steps_info else 0,
        "end_frame": steps_info[-1]['end_frame'] if steps_info else 0,
        "steps_info": steps_info
    }]

def default_serializer(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
