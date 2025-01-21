import math
import json
import random
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter


def plot_status(df, y_cols, x_axis_label=None, peaks_points=None):
    plt.figure(figsize=(30, 15), dpi=200)

    colors = ['blue', 'red', 'purple', 'orange', 'green', 'yellow']

    for idx, col in enumerate(y_cols):
        data = df[col]
        if x_axis_label:
            plt.plot(df[x_axis_label], data, label=col, marker='o', color=colors[idx % len(colors)])
            if peaks_points:
                plt.scatter(df[x_axis_label][peaks_points[idx]], data[peaks_points[idx]], color=random.choice(colors),
                            s=300)
        else:

            plt.plot(data, label=col, marker='x', color=colors[idx % len(colors)])

    title = ", ".join(y_cols) + f" over {x_axis_label}"
    plt.title(title.title(), fontsize=30)

    if x_axis_label:
        plt.xlabel(x_axis_label.title(), fontsize=20)
    else:
        plt.xlabel("X Axis", fontsize=20)

    if len(y_cols) == 1:
        plt.ylabel(y_cols[-1].title(), fontsize=20)
    else:
        plt.ylabel("Y Axis", fontsize=20)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()


def filter_outliers(df, z_scores_threshold=10, IQR_coefficient=7.5):
    z_scores = np.abs(stats.zscore(df))
    df_no_outliers_z_score = df[(z_scores < z_scores_threshold).all(axis=1)]
    Q1 = df_no_outliers_z_score.quantile(0.25)
    Q3 = df_no_outliers_z_score.quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = (df_no_outliers_z_score < (Q1 - IQR_coefficient * IQR)) | (
            df_no_outliers_z_score > (Q3 + IQR_coefficient * IQR))
    df_no_outliers = df_no_outliers_z_score[~outlier_mask.any(axis=1)]
    # reset index
    df_no_outliers.reset_index(drop=True, inplace=True)

    return df_no_outliers


def read_csv(csv_path):
    data = pd.read_csv(csv_path)
    no_outliers_data = filter_outliers(data)
    return no_outliers_data


def distance_foot(df):
    """
    calc the distance between left foot and right foot
    """
    df = df.copy()

    left_ankle = np.array(df.loc[:, ['left_ankle_x', 'left_ankle_y']])
    right_ankle = np.array(df.loc[:, ['right_ankle_x', 'right_ankle_y']])
    ankle_x_distance = np.abs(left_ankle[:, 0] - right_ankle[:, 0])
    ankle_y_distance = np.abs(left_ankle[:, 1] - right_ankle[:, 1])

    distance = np.linalg.norm(left_ankle - right_ankle, axis=1)

    df.loc[:, "foot_distance"] = distance
    df.loc[:, "ankle_x_distance"] = ankle_x_distance
    df.loc[:, "ankle_y_distance"] = ankle_y_distance

    return df


def smooth_data_for_finding_peaks(df, cols, sigma=2):
    for col in cols:
        df[f"{col}_smooth"] = gaussian_filter(df[col], sigma=sigma)

    return df


def find_ankle_peaks_valleys(df, distance=200, num_circle=3):
    left_ankle_x_peaks, _ = find_peaks(df["left_ankle_x_smooth"], distance=distance)
    right_ankle_x_peaks, _ = find_peaks(df["right_ankle_x_smooth"], distance=distance)
    # left_ankle_x_valleys, _ = find_peaks(-df["left_ankle_x_smooth"], distance=distance)
    # right_ankle_x_valleys, _ = find_peaks(-df["right_ankle_x_smooth"], distance=distance)

    n_circle = num_circle
    left_ankle_x_peaks_ = df["left_ankle_x_smooth"][left_ankle_x_peaks].tolist()
    left_ankle_x_peaks_.sort(reverse=True)
    right_ankle_x_peaks_ = df["right_ankle_x_smooth"][right_ankle_x_peaks].tolist()
    right_ankle_x_peaks_.sort(reverse=True)
    left_ankle_x_peaks_max = left_ankle_x_peaks_[:n_circle]
    right_ankle_x_peaks_max = right_ankle_x_peaks_[:n_circle]

    # left_ankle_x_valleys_ = distance_test1_data["left_ankle_x_smooth"][left_ankle_x_valleys].tolist()
    # left_ankle_x_valleys_.sort()
    # right_ankle_x_valleys_ = distance_test1_data["right_ankle_x_smooth"][right_ankle_x_valleys].tolist()
    # right_ankle_x_valleys_.sort()
    # left_ankle_x_valleys_min = left_ankle_x_valleys_[:n_circle]
    # right_ankle_x_valleys_min = right_ankle_x_valleys_[:n_circle]

    left_ankle_x_peaks_max_idx = df[
        df["left_ankle_x_smooth"].isin(left_ankle_x_peaks_max)].index.tolist()
    right_ankle_x_peaks_max_idx = df[
        df["right_ankle_x_smooth"].isin(right_ankle_x_peaks_max)].index.tolist()
    peaks_idx = [(l + r) // 2 for l, r in zip(left_ankle_x_peaks_max_idx, right_ankle_x_peaks_max_idx)]

    peaks_idx.insert(0, 0)
    peaks_idx.append(int(len(df) - 1))

    left_ankle_x_valleys_min_idx = []
    right_ankle_x_valleys_min_idx = []
    for i, idx in enumerate(peaks_idx):
        if i == len(peaks_idx) - 1:
            break
        left_valleys = df.iloc[idx: peaks_idx[i + 1], :].nsmallest(1, "left_ankle_x_smooth").index
        right_valleys = df.iloc[idx: peaks_idx[i + 1], :].nsmallest(1, "right_ankle_x_smooth").index
        left_ankle_x_valleys_min_idx.append(left_valleys[0])
        right_ankle_x_valleys_min_idx.append(right_valleys[0])

    valleys_idx = [(l + r) // 2 for l, r in zip(left_ankle_x_valleys_min_idx, right_ankle_x_valleys_min_idx)]
    peaks_idx = peaks_idx[1:-1]

    return peaks_idx, valleys_idx


def split_stages(peaks_idx, valleys_idx):
    peaks_idx.extend(valleys_idx)
    stage_idx = peaks_idx
    stage_idx.sort()

    return stage_idx


def get_n_circle_data(df, stage_idx, n):
    assert n < len(stage_idx)
    # if n < len(stage_idx):
    circle_n_data = df.iloc[stage_idx[n - 1]: stage_idx[n], :]

    return circle_n_data


def calculate_frame_len_avg(df, points):
    # calculate frame_len_avg
    frame_length = []
    for i in range(len(points)):
        if i == 0:
            data = df.iloc[:points[i], :]
            frame_len = data.iloc[-1, :]["frame_id"] - data.iloc[0, :]["frame_id"]
            frame_length.append(frame_len)
            data = df.iloc[points[i]:points[i + 1], :]
            frame_len = data.iloc[-1, :]["frame_id"] - data.iloc[0, :]["frame_id"]
            frame_length.append(frame_len)
        elif i == len(points) - 1:
            data = df.iloc[points[i]:, :]
        else:
            data = df.iloc[points[i]: points[i + 1], :]

        frame_len = data.iloc[-1, :]["frame_id"] - data.iloc[0, :]["frame_id"]
        frame_length.append(frame_len)

    return sum(frame_length) / len(frame_length)


def modified_peaks(df, peaks, valleys, threshold=0.6):
    peaks = list(peaks)
    valleys = list(valleys)

    # 校准波峰波谷
    # 根据new_peaks去df中的ankle_diff校验
    # 如果df["ankle_diff"][new_peaks[i]] < df["ankle_diff"][new_peaks[i] + 1]
    # 需要将i+1，直到df["ankle_diff"][new_peaks[i]] > df["ankle_diff"][new_peaks[i] + 1]
    for i, p in enumerate(peaks):
        while df["ankle_diff"][p] < df["ankle_diff"][p + 1] and p < len(df) - 1:
            peaks[i] = peaks[i] + 1
            p += 1
    for i, p in enumerate(valleys):
        while df["ankle_diff"][p] > df["ankle_diff"][p + 1] and p < len(df) - 1:
            valleys[i] = valleys[i] + 1
            p += 1

    peaks.extend(valleys)
    peaks.sort()
    distance = calculate_frame_len_avg(df, peaks)
    # print(peaks)

    new_peaks = []
    for p in peaks:
        if not new_peaks:
            new_peaks.append(p)
        else:
            if p - new_peaks[-1] > distance * threshold:
                new_peaks.append(p)

    return new_peaks


def calculate_step_length_speed(step_data, fps=60, ratio=0.00105):
    # calculate step_length
    left_ankle_x = step_data.iloc[-1, :]["left_ankle_x"]
    right_ankle_x = step_data.iloc[-1, :]["right_ankle_x"]

    # Direction
    if step_data.iloc[-1, :]["left_heel_x"] <= left_ankle_x:
        # --->
        front_leg = "left" if left_ankle_x > right_ankle_x else "right"
    else:
        front_leg = "left" if left_ankle_x < right_ankle_x else "right"

    step_length = abs(left_ankle_x - right_ankle_x) * ratio
    step_time = 1 / fps * (step_data.iloc[-1, :]['frame_id'] - step_data.iloc[0, :]['frame_id'])
    step_speed = step_length / step_time * ratio

    # print(
    #     f"Step info: {step_data.iloc[0, :]['frame_id']} - {step_data.iloc[-1, :]['frame_id']}, {front_leg}"
    #     f"step_length: {step_length:.2f}, step_speed: {step_speed} pixel/s")

    return step_length, step_speed, front_leg


def calculate_angle(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    v1 = (x2 - x1, y2 - y1)
    v2 = (x4 - x3, y4 - y3)

    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    magnitude_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = math.acos(cos_theta)
    angle_in_degrees = math.degrees(theta)

    return angle_in_degrees


def hip_degree(step_data):
    # if direction == 'right':
    #     swing_leg = 'left' if step_data.iloc[-1, :]['left_ankle_x'] > step_data.iloc[-1, :]['right_ankle_x'] else 'right'
    #     support_leg = 'right' if swing_leg == 'left' else 'left'
    # else:
    #     swing_leg = 'right' if step_data.iloc[-1, :]['left_ankle_x'] > step_data.iloc[-1, :]['right_ankle_x'] else 'left'
    #     support_leg = 'left' if swing_leg == 'right' else 'right'

    # calculate the degree between the swing leg and support leg
    v1_x = step_data["left_hip_x"] - step_data["left_knee_x"]
    v1_y = step_data["left_hip_y"] - step_data["left_knee_y"]
    v2_x = step_data["right_hip_x"] - step_data["right_knee_x"]
    v2_y = step_data["right_hip_y"] - step_data["right_knee_y"]

    dot_product = v1_x * v2_x + v1_y * v2_y
    v1_norm = np.sqrt(v1_x ** 2 + v1_y ** 2)
    v2_norm = np.sqrt(v2_x ** 2 + v2_y ** 2)
    cos_theta = dot_product / (v1_norm * v2_norm)
    cos_theta = cos_theta.clip(-1, 1)
    angle_radians = np.arccos(cos_theta)
    angle_degrees = np.degrees(angle_radians)

    # print(f"Step Hip Degree: {min(angle_degrees.tolist())} -- {max(angle_degrees.tolist())}")

    return min(angle_degrees.tolist()), max(angle_degrees.tolist())


def calculate_support_time(df, fps=60):
    diff2_ankle_smooth_mean = int(df["ankle_diff_smooth"].diff().mean())
    condition = df["ankle_diff_smooth"] < abs(diff2_ankle_smooth_mean)
    land_index = df[condition].index[-1] if condition.any() else None
    if land_index is None:
        land_index = df.index[-1]
    start_index = df.index[0]
    frame_n = int(df.loc[land_index, "frame_id"]) - int(df.loc[start_index, "frame_id"])
    support_time = 1 / fps * frame_n

    return support_time


def calculate_liftoff_height(df, front_leg, ratio=0.00105):
    back_leg = "right" if front_leg == "left" else "left"
    heightest = min(df.loc[:, f"{front_leg}_ankle_y"])
    lowest = max(df.loc[:, f"{back_leg}_ankle_y"])
    diff = lowest - heightest

    return diff * ratio


def analysis_circle(df, stage_idx, plot=False, fps=60, ratio=0.00105):
    df = smooth_data_for_finding_peaks(df, ["left_ankle_x", "right_ankle_x", "left_ankle_y", "right_ankle_y"], sigma=2)

    # if plot:
    #     for i, idx in enumerate(stage_idx[::2]):
    #         plot_status(distance_test1_data.iloc[idx:stage_idx[i + 1]],
    #         ['left_ankle_x_smooth', 'right_ankle_x_smooth'],
    #                     "frame_id")

    n = 1

    steps_detail = []
    while n < len(stage_idx):

        circle_n = df.iloc[stage_idx[n - 1]: stage_idx[n], :].copy()

        steps_dict = {
            "stage_n": n,
            "start_frame": circle_n.iloc[0, :]["frame_id"],
            "end_frame": circle_n.iloc[-1, :]["frame_id"],
            "steps_info": []
        }

        # circle_n["ankle_diff"] = circle_n["left_ankle_x"] - circle_n["right_ankle_x"]
        # circle_n["ankle_diff_smooth"] = circle_n["left_ankle_x_smooth"] - circle_n["right_ankle_x_smooth"]
        circle_n.loc[:, "ankle_diff"] = circle_n["left_ankle_x"] - circle_n["right_ankle_x"]
        circle_n.loc[:, "ankle_diff_smooth"] = circle_n["left_ankle_x_smooth"] - circle_n["right_ankle_x_smooth"]
        circle_n.reset_index(drop=True, inplace=True)

        peaks, _ = find_peaks(circle_n["ankle_diff_smooth"], distance=20)
        valleys, _ = find_peaks(-circle_n["ankle_diff_smooth"], distance=20)

        new_peaks = modified_peaks(circle_n, peaks, valleys)

        idx = 0
        # step_length_list = []
        # step_speed_list = []
        while idx < len(new_peaks):

            if idx == 0:
                step_data = circle_n.iloc[:new_peaks[idx], :]
                step_length, step_speed, front_leg = calculate_step_length_speed(step_data, fps=fps, ratio=ratio)
                support_time = calculate_support_time(step_data, fps=fps)
                liftoff_height = calculate_liftoff_height(df, front_leg, ratio)
                min_degree, max_degree = hip_degree(step_data)

                steps_dict["steps_info"].append(
                    {
                        "start_frame": step_data.iloc[0, :]["frame_id"],
                        "end_frame": step_data.iloc[-1, :]["frame_id"],
                        "step_length": step_length,
                        "step_speed": step_speed,
                        "front_leg": front_leg,
                        "support_time": support_time,
                        "liftoff_height": liftoff_height,
                        "hip_min_degree": min_degree,
                        "hip_max_degree": max_degree,
                        "first_step": True,
                        "steps_diff": -1,
                        "stride_length": -1
                    }
                )

                step_data = circle_n.iloc[new_peaks[idx]:new_peaks[idx + 1], :]
                step_length, step_speed, front_leg = calculate_step_length_speed(step_data, fps=fps, ratio=ratio)
                support_time = calculate_support_time(step_data, fps=fps)
                liftoff_height = calculate_liftoff_height(df, front_leg, ratio)
                min_degree, max_degree = hip_degree(step_data)

            elif idx == len(new_peaks) - 1:
                step_data = circle_n.iloc[new_peaks[idx]:, :]
                step_length, step_speed, front_leg = calculate_step_length_speed(step_data, fps=fps, ratio=ratio)
                support_time = calculate_support_time(step_data, fps=fps)
                liftoff_height = calculate_liftoff_height(df, front_leg, ratio)
                min_degree, max_degree = hip_degree(step_data)

            else:
                step_data = circle_n.iloc[new_peaks[idx]: new_peaks[idx + 1], :]
                step_length, step_speed, front_leg = calculate_step_length_speed(step_data, fps=fps, ratio=ratio)
                support_time = calculate_support_time(step_data, fps=fps)
                liftoff_height = calculate_liftoff_height(df, front_leg, ratio)
                min_degree, max_degree = hip_degree(step_data)

            steps_dict["steps_info"].append(
                {
                    "start_frame": step_data.iloc[0, :]["frame_id"],
                    "end_frame": step_data.iloc[-1, :]["frame_id"],
                    "step_length": step_length,
                    "step_speed": step_speed,
                    "front_leg": front_leg,
                    "support_time": support_time,
                    "liftoff_height": liftoff_height,
                    "hip_min_degree": min_degree,
                    "hip_max_degree": max_degree,
                    "first_step": False,
                    "steps_diff": (step_length - steps_dict["steps_info"][-1]["step_length"]) * ratio,
                    "stride_length": (step_length + steps_dict["steps_info"][-1]["step_length"]) * ratio
                }
            )
            idx += 1

        steps_detail.append(steps_dict)
        n += 1

    return steps_detail


def main_analysis(csv, smooth_sigma=20, num_circle=3, stage_gap_dis=200):
    data = read_csv(csv)
    distance_data = distance_foot(data)
    smooth_data = smooth_data_for_finding_peaks(distance_data, ["left_ankle_x", "right_ankle_x"], sigma=smooth_sigma)
    peaks, valleys = find_ankle_peaks_valleys(smooth_data, distance=stage_gap_dis, num_circle=num_circle)
    stage_idx = split_stages(peaks, valleys)
    step_detail = analysis_circle(smooth_data, stage_idx)

    with open(csv.replace(".csv", "_results.json"), "w") as o:
        json.dump(step_detail, o, indent=4)

    # print(step_detail)
    return step_detail


if __name__ == '__main__':
    main_analysis("72_raw.csv")
