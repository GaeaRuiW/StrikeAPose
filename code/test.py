from ultralytics import YOLO
import cv2

# 加载预训练的YOLOv8姿态估计模型
model = YOLO("yolov8x-pose.pt")  # 使用您下载的模型路径

# 从视频中读取一帧画面
cap = cv2.VideoCapture("test.mp4")  # 替换为您的视频路径
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)  # 设置读取第100帧
ret, frame = cap.read()

# 进行推理
results = model.predict(frame, save=True, imgsz=640, conf=0.5, iou=0.5)

# 获取第一个结果（因为我们只处理一帧）
result = results[0]

# 初始化数据结构
detections = []

# 遍历所有检测到的对象
for i, box in enumerate(result.boxes):
    # 获取边界框坐标 (xyxy格式: x_min, y_min, x_max, y_max)
    bbox = box.xyxy[0].tolist()
    
    # 获取置信度
    confidence = box.conf[0].item()
    
    # 获取类别ID和名称
    cls_id = int(box.cls[0].item())
    cls_name = result.names[cls_id]
    
    # 获取对应的关键点（如果存在）
    keypoints = None
    if result.keypoints is not None and i < len(result.keypoints):
        kpts = result.keypoints[i].xy[0].tolist()  # 获取所有关键点坐标
        
        # 创建关键点名称映射（COCO格式）
        keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        # 构建关键点字典 {名称: (x, y)}
        keypoints = {name: (x, y) for name, (x, y) in zip(keypoint_names, kpts)}
    
    # 构建检测对象信息
    detection = {
        "object_id": i + 1,
        "class_id": cls_id,
        "class_name": cls_name,
        "confidence": round(confidence, 4),
        "bounding_box": {
            "x_min": round(bbox[0], 2),
            "y_min": round(bbox[1], 2),
            "x_max": round(bbox[2], 2),
            "y_max": round(bbox[3], 2),
            "width": round(bbox[2] - bbox[0], 2),
            "height": round(bbox[3] - bbox[1], 2)
        },
        "keypoints": keypoints
    }
    
    detections.append(detection)

# 打印结果
print(f"检测到 {len(detections)} 个对象:")
for i, det in enumerate(detections):
    print(f"\n对象 #{i+1}:")
    print(f"  类别: {det['class_name']} (置信度: {det['confidence']})")
    print(f"  边界框: [x1={det['bounding_box']['x_min']}, y1={det['bounding_box']['y_min']}, "
          f"x2={det['bounding_box']['x_max']}, y2={det['bounding_box']['y_max']}]")
    
    if det['keypoints']:
        print("  关键点:")
        for kp_name, (x, y) in det['keypoints'].items():
            print(f"    {kp_name}: ({round(x, 2)}, {round(y, 2)})")

# 可视化结果（可选）
# annotated_frame = result.plot()  # 绘制检测结果

# 显示带注释的图像
# cv2.imshow("Pose Detection Results", annotated_frame)
# cv2.waitKey(0)  # 按任意键关闭窗口
# cv2.destroyAllWindows()

# 释放视频资源
cap.release()

# 保存结果到文件（可选）
# import json
# with open("detection_results.json", "w") as f:
#     json.dump(detections, f, indent=2)

# print("\n结果已保存到 detection_results.json")