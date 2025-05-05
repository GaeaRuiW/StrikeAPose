import os
import subprocess
import statistics

import bcrypt
import cv2
import redis
from config import length_to_show, redis_db, redis_host, redis_port


def get_redis_connection():
    return redis.Redis(host=redis_host, port=redis_port, db=redis_db)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def check_password(password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed_password.encode())


def get_length_to_show():
    return os.getenv("LENGTH_TO_SHOW", length_to_show)


def calculate_stats(data: list[float | int | None]) -> tuple[float, float]:
    filtered_data = [x for x in data if x is not None]

    n = len(filtered_data)
    if n == 0:
        return 0.0, 0.0

    average = sum(filtered_data) / n

    if n < 2:
        std_dev = 0.0
    else:
        std_dev = statistics.stdev(filtered_data)

    return round(average, 2), round(std_dev, 2)


def generate_thumbnail(video_path: str, thumbnail_path: str, time: int = 1):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, time * 1000)
    success, image = cap.read()
    if success:
        cv2.imwrite(thumbnail_path, image)
    cap.release()

def convert_to_mp4(input_path, output_path, ffmpeg_executable="ffmpeg"):
    if not os.path.exists(input_path):
        print(f"错误：输入文件不存在: {input_path}")
        return False

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"已创建输出目录: {output_dir}")
        except OSError as e:
            print(f"错误：无法创建输出目录 {output_dir}: {e}")
            return False

    command = [
        ffmpeg_executable,
        "-i", input_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-y",
        output_path
    ]

    print(f"\n正在转换: {input_path} -> {output_path}")
    print(f"执行命令: {' '.join(command)}")

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("转换成功!")
        return True
    except FileNotFoundError:
        print(f"错误：找不到 FFmpeg 可执行文件 '{ffmpeg_executable}'。请检查路径或 PATH 设置。")
        return False
    except subprocess.CalledProcessError as e:
        print(f"错误：FFmpeg 转换失败 (返回码: {e.returncode})")
        print("FFmpeg 标准输出:\n", e.stdout)
        print("FFmpeg 标准错误:\n", e.stderr) # 错误信息通常在这里
        # 如果转换失败，尝试删除可能已创建的不完整输出文件
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                print(f"已删除不完整的输出文件: {output_path}")
            except OSError as remove_err:
                print(f"警告：无法删除不完整的输出文件 {output_path}: {remove_err}")
        return False
    except Exception as e:
        print(f"发生未知错误: {e}")
        return False