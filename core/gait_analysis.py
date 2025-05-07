import numpy as np
from collections import defaultdict

# 计算角度
def calculate_angle(a, b, c):
    """计算三点之间的夹角（单位：弧度）"""
    # 将a、b、c转换为numpy数组
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    # 计算向量ba和bc的点积
    dot_product = np.dot(ba, bc)
    # 计算向量ba和bc的模
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    # 如果向量ba或bc的模为0，则返回0
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    # 计算向量ba和bc的夹角的余弦值
    cosine_angle = dot_product / (norm_ba * norm_bc)
    # 将余弦值限制在-1到1之间
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    # 计算弧度后转换为角度
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

# 计算步态参数
def calculate_gait_parameters(left_points, right_points, smoothed_data, left_turn, right_turn, fps=24.0):
    print(left_turn,right_turn)
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
    
    '''
    # 计算步宽
    left_land_ys = [step['land_y'] for step in all_steps if step['leg'] == 'left']
    right_land_ys = [step['land_y'] for step in all_steps if step['leg'] == 'right']
    avg_left_y = np.mean(left_land_ys) if left_land_ys else 0
    avg_right_y = np.mean(right_land_ys) if right_land_ys else 0
    step_width_value = abs(avg_left_y - avg_right_y)
    '''

    # 获取首尾关键点
    first_step = all_steps[1]
    last_step = all_steps[-2]
    # 计算总位移（取绝对值）
    start_x = first_step['lift_x']  # 第一个离地点x坐标
    end_x = last_step['land_x']     # 最后一个着地点x坐标
    
    if left_turn or right_turn:
        left_ankle = [d['left_ankle'] for d in smoothed_data][left_turn][0]
        right_ankle = [d['right_ankle'] for d in smoothed_data][right_turn][0]
        mid_x = left_ankle if left_ankle > right_ankle else right_ankle  # 中间点x坐标
        total_distance = abs(end_x - mid_x) + abs(mid_x - start_x)
    else:
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
    pre_step_length = None

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
                last_left_land_y = land_y
                
            else:
                last_right_land_x = land_x
                last_right_land_y = land_y
            continue
        
        # 计算步宽
        if leg == 'left':
            if last_right_land_y is not None:
                step_width_value = abs(land_y - last_right_land_y)
            else:
                step_width_value = 0
            last_left_land_y = land_y  # 更新左脚的着地点
        else:
            if last_left_land_y is not None:
                step_width_value = abs(land_y - last_left_land_y)
            else:
                step_width_value = 0
            last_right_land_y = land_y  # 更新右脚的着地点

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

        # 计算步长差
        if pre_step_length is not None:
            steps_diff = abs(step_length - pre_step_length)
        else:
            steps_diff = 0
        pre_step_length = step_length

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
            ankle_ys = [d['left_ankle_z'][1] for d in frames_in_step]
        else:
            ankle_ys = [d['right_ankle_z'][1] for d in frames_in_step]
        if ankle_ys:
            max_y = max(ankle_ys)
            min_y = min(ankle_ys)
            print ("ankle_ys:",ankle_ys)
            print("max:",max_y)
            print("min:",min_y)
            liftoff_height = (max_y - min_y)
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
            'steps_diff': steps_diff,
            'stride_length': stride_length
        }

        '''
        # 如果存在前一步，计算并更新前一步的步长差
        if prev_step_info is not None:
            steps_diff = abs(current_step_length - prev_step_info['step_length'])
            prev_step_info['steps_diff'] = steps_diff
        '''
        steps_info.append(step_info)
        # prev_step_info = step_info  # 更新前一步为当前步骤
        
    
    # steps_info = steps_info[1:-1]

    # 过滤转身脚步
    if left_turn or right_turn:
        exclude_indices = set()
        # 情况1：排除包含转身点的步骤
        for i, step in enumerate(steps_info):
            front_leg = step['front_leg']
            start = step['start_frame']
            end = step['end_frame']
            if (front_leg == 'left' and start <= left_turn <= end) or \
            (front_leg == 'right' and start <= right_turn <= end):
                exclude_indices.add(i)

        # 情况2：处理转身点位于步骤间隙的情况
        if not exclude_indices:
            # 按时间排序步骤并记录原始索引
            sorted_indices = sorted(range(len(steps_info)), 
                                key=lambda i: steps_info[i]['start_frame'])
            sorted_steps = [steps_info[i] for i in sorted_indices]
            
            # 检查所有转身点
            for turn_point in [left_turn, right_turn]:
                # 遍历寻找包含转身点的间隙
                for i in range(len(sorted_steps) - 1):
                    prev_step = sorted_steps[i]
                    next_step = sorted_steps[i+1]
                    # 检查间隙条件
                    if prev_step['end_frame'] < turn_point < next_step['start_frame']:
                        # 获取原始索引
                        prev_idx = sorted_indices[i]
                        next_idx = sorted_indices[i+1]
                        exclude_indices.update({prev_idx, next_idx})

        # 排除相邻步骤（前一项和后一项）
        additional_exclude = set()
        for i in exclude_indices:
            if i - 1 >= 0:
                additional_exclude.add(i - 1)
            if i + 1 < len(steps_info):
                additional_exclude.add(i + 1)
        
        all_exclude = exclude_indices.union(additional_exclude)

        # 应用过滤
        filtered_steps = [step for i, step in enumerate(steps_info) if i not in all_exclude]
        steps_info = filtered_steps

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