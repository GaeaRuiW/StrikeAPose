import os
import sys
import time

import cv2
import numpy as np
import torch
from check_point import caculate_output, get_featurepoints2
from ultralytics import YOLO

# 13-16 左膝 右膝 左踝 右踝


def main(input_vedio_file, out_video_file, out_json_file):
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    model_path = current_dir + '/yolov8x-pose.pt'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0
    video = cv2.VideoWriter(out_video_file, fourcc, fps, (1280, 720))
    model = YOLO(model_path)
    cap = cv2.VideoCapture(input_vedio_file)
    data = []
    keypoint_ret = False
    key_points = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            if not keypoint_ret:
                keypoint_ret, key_points = get_featurepoints2(frame)
                if not keypoint_ret:
                    keypoint_ret = 2

            results = model.predict(frame, conf=0.8, iou=0.2, verbose=False)[0]
            keypoints_ = results.keypoints.xy.cpu().numpy()[0]
            keypoints = keypoints_[-6:, :]

            if keypoint_ret != 2:
                cv2.line(frame, key_points[0], key_points[1], (0, 255,
                         255), thickness=2, lineType=cv2.LINE_8, shift=0)
                cv2.line(frame, key_points[1], key_points[3], (0, 255,
                         255), thickness=2, lineType=cv2.LINE_8, shift=0)
                cv2.line(frame,  key_points[3],  key_points[2], (0,
                         255, 255), thickness=2, lineType=cv2.LINE_8, shift=0)
                cv2.line(frame,  key_points[2],  key_points[0], (0,
                         255, 255), thickness=2, lineType=cv2.LINE_8, shift=0)

            if len(keypoints_) > 0:
                for i in range(len(keypoints_)):
                    if i == 13 or i == 15:
                        cv2.circle(frame, (int(keypoints_[i, 0]), int(
                            keypoints_[i, 1])), 2, (0, 0, 255), 2)
                    elif i == 14 or i == 16:
                        cv2.circle(frame, (int(keypoints_[i, 0]), int(
                            keypoints_[i, 1])), 2, (0, 255, 0), 2)
                    if i < 13:
                        cv2.circle(frame, (int(keypoints_[i, 0]), int(
                            keypoints_[i, 1])), 2, (255, 0, 0), 2)
                cv2.line(frame, (int(keypoints_[13, 0]), int(keypoints_[13, 1])), (int(keypoints_[15, 0]),
                                                                                   int(keypoints_[15, 1])), (0, 0, 255), thickness=1, lineType=cv2.LINE_8, shift=0)
                cv2.line(frame, (int(keypoints_[14, 0]), int(keypoints_[14, 1])), (int(keypoints_[16, 0]),
                                                                                   int(keypoints_[16, 1])), (0, 255, 0), thickness=1, lineType=cv2.LINE_8, shift=0)
                data.append(keypoints)
            video.write(frame)
    video.release()
    cv2.destroyAllWindows()
    if keypoint_ret == 2:
        print("No json data")
    else:
        os.system(
            f"ffmpeg -i {out_video_file} -c:v libx264 -c:a aac {out_video_file.replace('.mp4', '_final.mp4')}")
        os.system(
            f"mv -f {out_video_file.replace('.mp4', '_final.mp4')} {out_video_file}")
        return caculate_output(key_points, np.array(data), out_json_file)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"command 1.inference 2.input_vedio_file,3.out_video_file,4.out_json_file")
        sys.exit(-1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
    # for index in range(1,10):
    #     print("index {}".format(index))
    #     data_in = ["data/{}.mp4".format(index), "data/{}out.mp4".format(index), "data/{}out.json".format(index)]
    #     main(data_in[0],data_in[1],data_in[2])
