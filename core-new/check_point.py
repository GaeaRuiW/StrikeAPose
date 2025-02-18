import copy
import json
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np


def default_serializer(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    raise TypeError(
        f"Object of type {obj.__class__.__name__} is not JSON serializable")


def diff_non_zero(arr):
    # 过滤出非零值
    non_zero_values = [i for i in arr if i != 0]
    if not non_zero_values:
        return 0
    # 计算非零值中的最大值和最小值的差
    return max(non_zero_values) - min(non_zero_values)


def sort_points(points):

    points_sorted_by_x = sorted(points, key=lambda p: p[0])

    x_first_two = points_sorted_by_x[:2]
    x_last_two = points_sorted_by_x[2:]
    x_first_two_sorted = sorted(x_first_two, key=lambda p: p[1])
    x_last_two_sorted = sorted(x_last_two, key=lambda p: p[1])
    sorted_points = [
        x_first_two_sorted[0],
        x_last_two_sorted[0],
        x_first_two_sorted[1],
        x_last_two_sorted[1]
    ]

    return sorted_points


def get_featurepoints2(image0):
    image = image0.copy()
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_color = np.array([0, 80, 80])
    upper_color = np.array([30, 255, 255])
    mask1 = cv2.inRange(image1, lower_color, upper_color)

    lower_color = np.array([150, 80, 80])
    upper_color = np.array([180, 255, 255])
    mask2 = cv2.inRange(image1, lower_color, upper_color)

    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened_image = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ellipses = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if len(contour) > 5 and area > 600:

            ellipse = cv2.fitEllipse(contour)
            (center, axes, angle) = ellipse
            ellipses.append((center, ellipse))

    ellipses.sort(key=lambda x: x[0][1], reverse=True)
    top_ellipses = ellipses[:4]
    key_point = []
    if len(top_ellipses) == 4:
        for center, ellipse in top_ellipses:
            cv2.ellipse(image, ellipse, (0, 255, 0), 2)
            key_point.append((int(center[0]), int(center[1])))
        key_point = sort_points(key_point)
        print("check 4 points sucess !")
        return True, key_point
    else:
        print("check 4 points fail !")
        return False, key_point


def caculate_output(key_points4, datas, out_file):
    box_len = 3.0
    box_width = 0.8
    fps = 24.0
    left_top_point = [key_points4[0][0], key_points4[0][1]]
    right_top_point = [key_points4[1][0], key_points4[1][1]]
    left_bottom_point = [key_points4[2][0], key_points4[2][1]]
    right_bottom_point = [key_points4[3][0], key_points4[3][1]]
    pixel_one = right_bottom_point[0] - left_bottom_point[0]
    pixel_two = right_top_point[0] - left_top_point[0]
    erro = (pixel_one - pixel_two) / 4
    p1 = pixel_one - abs(erro)
    p2 = pixel_two + abs(erro)
    data = datas[:, -4:, ]
    l_data = len(data) - 1
    i = 0
    while i < l_data:
        i += 1
        for j in range(2, 4):
            if data[i, j, 0] == 0:
                data[i, j, 0] = data[i - 1, j, 0]
    index_max_left = np.argmax(data[:, 2, 0])
    index_max_right = np.argmax(data[:, 3, 0])
    index_max_all = 0
    left1 = []
    left2 = []
    right1 = []
    right2 = []
    i = 0
    while i < l_data:
        i += 1
        if data[i - 1, 2, 0] > data[i - 1, 3, 0] and data[i, 2, 0] < data[i, 3, 0]:
            if i < index_max_left:
                left1.append(i)
            else:
                right2.append(i)
            i += 10
    i = 0
    while i < l_data:
        i += 1
        if data[i - 1, 2, 0] < data[i - 1, 3, 0] and data[i, 2, 0] > data[i, 3, 0]:
            if i < index_max_right:
                right1.append(i)
            else:
                left2.append(i)
            i += 10

    left1_mean = left1[-2] - left1[-3]
    for i in range(len(left1)):
        if left1[-i] - left1[-i - 1] > left1_mean * 1.5:
            left1 = left1[-i:]
            break
    right1_mean = right1[-2] - right1[-3]
    for i in range(len(right1)):
        if right1[-i] - right1[-i - 1] > right1_mean * 1.5:
            right1 = right1[-i:]
            break
    start = int(max(left1[0], right1[0]) - 1.5 * max(left1_mean, right1_mean))

    left2_mean = (left2[-1] - left2[0]) / (len(left2) - 1)
    for i in range(len(left2) - 1):
        if left2[i + 1] - left2[i] > left2_mean * 1.5:
            left2 = left2[:i + 1]
            break

    right2_mean = (right2[-1] - right2[0]) / (len(right2) - 1)
    for i in range(len(right2) - 1):
        if right2[i + 1] - right2[i] > right2_mean * 1.5:
            right2 = right2[:i + 1]
            break
    end = int(min(right2[-1], left2[-1]) + 1.0 * max(left2_mean, right2_mean))

    if left1[0] < right1[0] and abs(data[left1[0], 2, 0] - data[start, 2, 0]) < (left2_mean + left1_mean) / 4:
        left1 = left1[1:]
    if left1[0] > right1[0] and abs(data[right1[0], 3, 0] - data[start, 3, 0]) < (right2_mean + right1_mean) / 4:
        right1 = right1[1:]

    if left2[-1] > right2[-1] and abs(data[left2[-1], 2, 0] - data[end, 2, 0]) < (left2_mean + left1_mean) / 4:
        left2 = left2[:-1]
    if left2[-1] < right2[-1] and abs(data[right2[-1], 3, 0] - data[end, 3, 0]) < (right2_mean + right1_mean) / 4:
        right2 = right2[:-1]

    if data[index_max_left, 2, 0] > data[index_max_right, 3, 0]:
        index_max_all = index_max_left
        left = [start] + left1 + [index_max_left] + left2 + [int(end)]
        # if abs(data[right1[-1], 3, 0] - data[right2[0], 3, 0]) < (right2_mean + right1_mean) / 3:
        if right1[-1] > left1[-1] and right2[0] < left2[0]:
            right = [start] + right1[:-1] + \
                [int((right1[-1] + right2[0]) / 2)] + right2[1:] + [int(end)]
        else:  # 正常清况
            right = [start] + right1 + right2 + [int(end)]
    else:
        index_max_all = index_max_right
        right = [start] + right1 + [index_max_right] + right2 + [int(end)]
        # if abs(data[left1[-1], 2, 0] - data[left2[0], 2, 0]) < (left2_mean + left1_mean) / 3:
        if left1[-1] > right1[-1] and left2[0] < right2[0]:
            left = [start] + left1[:-1] + \
                [int((left1[-1] + left2[0]) / 2)] + left2[1:] + [int(end)]
        else:  # 正常情况
            left = [start] + left1 + left2 + [int(end)]

    # divide left and right
    left_step_move = []
    for i in range(len(left) - 1):
        a = left[i]
        b = left[i + 1]
        while a < b - 10:
            if abs(data[a, 2, 0] - data[a + 10, 2, 0]) > 65:
                for k in range(9):
                    if (abs(data[a + k, 2, 0] - data[a + k + 1, 2, 0]) > 5
                            and abs(data[a + k + 1, 2, 0] - data[a + k + 2, 2, 0]) > 5
                            and (data[a + k, 2, 0] - data[a + k + 1, 2, 0]) * (
                            data[a + k + 1, 2, 0] - data[a + k + 2, 2, 0]) > 0):
                        break
                a += k
                break
            a += 1
        while b > a + 10:
            if abs(data[b, 2, 0] - data[b - 10, 2, 0]) > 65:
                for k in range(9):
                    if (abs(data[b - k, 2, 0] - data[b - k - 1, 2, 0]) > 5
                            and abs(data[b - k - 1, 2, 0] - data[b - k - 2, 2, 0]) > 5
                            and (data[b - k, 2, 0] - data[b - k - 1, 2, 0]) * (
                            data[b - k - 1, 2, 0] - data[b - k - 2, 2, 0]) > 0):
                        break
                b -= k
                break
            b -= 1
        left_step_move.append([a, b])
    right_step_move = []
    for i in range(len(right) - 1):
        a = right[i]
        b = right[i + 1]
        while a < b - 10:
            if abs(data[a, 3, 0] - data[a + 10, 3, 0]) > 65:
                for k in range(9):
                    if ((abs(data[a + k, 3, 0] - data[a + k + 1, 3, 0]) > 5
                         and abs(data[a + k + 1, 3, 0] - data[a + k + 2, 3, 0]) > 5)
                            and (data[a + k, 3, 0] - data[a + k + 1, 3, 0]) * (
                            data[a + k + 1, 3, 0] - data[a + k + 2, 3, 0]) > 0):
                        break
                a += k
                break
            a += 1
        while b > a + 10:
            if abs(data[b, 3, 0] - data[b - 10, 3, 0]) > 65:
                for k in range(9):
                    if (abs(data[b - k, 3, 0] - data[b - k - 1, 3, 0]) > 5
                            and abs(data[b - k - 1, 3, 0] - data[b - k - 2, 3, 0]) > 5
                            and (data[b - k, 3, 0] - data[b - k - 1, 3, 0]) * (
                            data[b - k - 1, 3, 0] - data[b - k - 2, 3, 0]) > 0):
                        break
                b -= k
                break
            b -= 1
        right_step_move.append([a, b])

    all_step = left_step_move
    all_leg = ["left" for i in range(len(left_step_move))]
    k = 0
    for step in right_step_move:
        for i in range(k, len(all_step), 1):
            if step[0] < all_step[i][0]:
                all_step.insert(i, step)
                all_leg.insert(i, "right")
                k = i
                break
            if i == len(all_step):
                all_step.append(step)
                all_leg.append("right")

    all_step_ = copy.deepcopy(all_step)
    for i in range(len(all_step_) - 1):
        value = (all_step_[i][1] + all_step_[i + 1][0]) / 2
        all_step_[i][1] = int(value)
        all_step_[i + 1][0] = all_step_[i][1] + 1

    data0 = datas[:, :4, :]
    angles = []
    for i in range(data0.shape[0]):
        vector1 = data0[i, 0] - data0[i, 2]
        vector2 = data0[i, 1] - data0[i, 3]
        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)
        cos_angle = dot_product / (magnitude1 * magnitude2)
        angle_rad = np.arccos(cos_angle)
        angles.append(angle_rad)

    steps_info = [[], []]
    start_frame_list = [all_step_[0][0], all_step_[0][0]]
    end_frame_list = [all_step_[-1][1], all_step_[-1][1]]
    direction_key = 0
    last_step_length = 0
    for i in range(len(all_step)):
        one_step_info = {}
        start_frame = all_step_[i][0]
        end_frame = all_step_[i][1]
        front_leg = all_leg[i]
        step_length = 0
        liftoff_height = 0
        step_width = 0
        stride_length = 0

        first_step = False
        if i == 0:
            first_step = True
        if i > 0 and direction_key == 0 and end_frame > index_max_all:
            direction_key = 1
            first_step = True
            start_frame_list[1] = start_frame
            end_frame_list[0] = start_frame - 1

        if front_leg == "left":
            step_length = data[end_frame, 2, 0] - data[end_frame, 3, 0]
            step_width = data[end_frame, 2, 1] - data[end_frame, 3, 1]
            if step_length > 0:
                step_length = abs(step_length) * box_len / p1
                liftoff_height = diff_non_zero(
                    data[start_frame:end_frame, 2, 1]) * box_len / p1
                step_width = abs(step_width) * box_len / p1
            else:
                step_length = abs(step_length) * box_len / p2
                liftoff_height = diff_non_zero(
                    data[start_frame:end_frame, 2, 1]) * box_len / p2
                step_width = abs(step_width) * box_len / p2
            stride_length = data[end_frame, 2, 0] - data[start_frame, 2, 0]
            if stride_length > 0:
                stride_length = abs(stride_length) * box_len / p1
            else:
                stride_length = abs(stride_length) * box_len / p2

        if front_leg == "right":
            step_length = data[end_frame, 3, 0] - data[end_frame, 2, 0]
            step_width = data[end_frame, 3, 1] - data[end_frame, 2, 1]
            if step_length > 0:
                step_length = abs(step_length) * box_len / p1
                liftoff_height = diff_non_zero(
                    data[start_frame:end_frame, 3, 1]) * box_len / p1
                step_width = abs(step_width) * box_len / p1
            else:
                step_length = abs(step_length) * box_len / p2
                liftoff_height = diff_non_zero(
                    data[start_frame:end_frame, 3, 1]) * box_len / p2
                step_width = abs(step_width) * box_len / p2
            stride_length = data[end_frame, 3, 0] - data[start_frame, 3, 0]
            if stride_length > 0:
                stride_length = abs(stride_length) * box_len / p1
            else:
                stride_length = abs(stride_length) * box_len / p2

        support_time = (all_step[i][1] - all_step[i][0]) / fps
        step_speed = step_length / (all_step_[i][1] - all_step_[i][0]) * fps

        hip_min_degree = min(angles[start_frame:end_frame])
        hip_max_degree = max(angles[start_frame:end_frame])
        if last_step_length > 0:
            steps_diff = step_length - last_step_length
        else:
            steps_diff = -1
        last_step_length = step_length

        one_step_info["start_frame"] = start_frame
        one_step_info["end_frame"] = end_frame
        one_step_info["step_length"] = step_length
        one_step_info["step_width"] = float(step_width)
        one_step_info["step_speed"] = step_speed
        one_step_info["front_leg"] = front_leg
        one_step_info["support_time"] = support_time
        one_step_info["liftoff_height"] = liftoff_height
        one_step_info["hip_min_degree"] = float(hip_min_degree)
        if np.isnan(float(hip_min_degree)):
            one_step_info["hip_min_degree"] = 0
        one_step_info["hip_max_degree"] = float(hip_max_degree)
        if np.isnan(float(hip_max_degree)):
            one_step_info["hip_max_degree"] = 0

        one_step_info["first_step"] = first_step
        one_step_info["steps_diff"] = steps_diff
        one_step_info["stride_length"] = float(stride_length)

        steps_info[direction_key].append(one_step_info)

    out_info = []
    for i in range(2):
        out_info_one_stage = {}
        out_info_one_stage["stage_n"] = i + 1
        out_info_one_stage["start_frame"] = start_frame_list[i]
        out_info_one_stage["end_frame"] = end_frame_list[i]
        out_info_one_stage["steps_info"] = steps_info[i]
        out_info.append(out_info_one_stage)

    with open(out_file, 'w') as json_file:
        json.dump(out_info, json_file, indent=4, default=default_serializer)
    print("end!")


if __name__ == "__main__":
    image = cv2.imread("image_gride_new5.jpg")
    get_featurepoints2(image)
