from ultralytics import YOLO

# 加载您的姿态估计模型
print("正在导出姿态模型 yolov8x-pose.pt ...")
pose_model = YOLO('yolov8x-pose.pt')
# 以半精度(FP16)导出为TensorRT引擎
# 这会生成一个 yolov8x-pose.engine 文件
pose_model.export(format='engine', half=True)
print("姿态模型导出完成！-> yolov8x-pose.engine")

print("-" * 20)

# 加载您的标定点检测模型
print("正在导出标定点模型 yolov8s-point.pt ...")
point_model = YOLO('yolov8s-point.pt')
# 以半精度(FP16)导出为TensorRT引擎
# 这会生成一个 yolov8s-point.engine 文件
point_model.export(format='engine', half=True)
print("标定点模型导出完成！ -> yolov8s-point.engine") 