import cv2

def extract_frames(input_video_path, output_video_path, num_frames=100):
    """
    从视频中提取指定数量的帧并保存为新视频
    
    参数:
    input_video_path: 输入视频文件路径
    output_video_path: 输出视频文件路径
    num_frames: 要提取的帧数量 (默认为100)
    """
    # 打开视频文件
    cap = cv2.VideoCapture(input_video_path)
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {input_video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息: {width}x{height}, {fps:.2f} FPS, 总帧数: {total_frames}")
    
    # 确定实际要提取的帧数
    frames_to_extract = min(num_frames, total_frames)
    print(f"正在提取前 {frames_to_extract} 帧...")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编码
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 读取并写入帧
    frame_count = 0
    while frame_count < frames_to_extract:
        ret, frame = cap.read()
        
        if not ret:
            print(f"警告: 在帧 {frame_count} 处提前结束")
            break
        
        # 写入帧到输出视频
        out.write(frame)
        frame_count += 1
        
        # 每10帧打印一次进度
        if frame_count % 10 == 0:
            print(f"已处理 {frame_count}/{frames_to_extract} 帧")
    
    # 释放资源
    cap.release()
    out.release()
    
    print(f"\n成功保存 {frame_count} 帧到 {output_video_path}")

# 使用示例
if __name__ == "__main__":
    input_video = "test2.mp4"  # 替换为您的输入视频路径
    output_video = "output2_100_frames.mp4"  # 输出视频路径
    
    extract_frames(input_video, output_video, num_frames=100)