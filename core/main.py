import argparse
import warnings
from rtmpose_trt_inference import *
from steps_analysis import *

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    """
    --video_path /home/zhaosilei/Projects/PycharmProjects/mmlab/mmpose-1.3.1/tools/workdirs/rtmpose-m_coco25/steps_analysis/73_raw.MP4
    --output_path /home/zhaosilei/Projects/PycharmProjects/mmlab/mmpose-1.3.1/tools/workdirs/rtmpose-m_coco25/output
    --diff 1
    --num_circle 3
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='')
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--diff', type=str, default=3)
    parser.add_argument('--num_circle', type=str, default=3)
    parser.add_argument('--smooth_sigma', type=str, default=15)
    parser.add_argument('--vis', action='store_true')

    args = parser.parse_args()
    video_path = args.video_path
    output_path = args.output_path
    differece_frames = int(args.diff)
    num_circle = int(args.num_circle)
    smooth_sigma = int(args.smooth_sigma)
    vis = True if args.vis else False

    result = inference(video_path, output_path, differece_frames, smooth_sigma, num_circle, vis)
    if result is not None:
        print(result)
    else:
        print('No result! please check the csv and log file.')