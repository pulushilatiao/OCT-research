from ultralytics import YOLO

# 加载训练好的 YOLOv8 模型
model = YOLO("yolov8n.pt")

# 导出为 ONNX 格式
# dynamic=False: 嵌入式端建议固定 Shape，推理性能高于动态 Shape
# opset=12: 确保算子兼容性良好
model.export(
    format="onnx",
    imgsz=[640, 640],
    dynamic=False,
    simplify=True,
    opset=12
)