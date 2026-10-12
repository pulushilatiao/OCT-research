import os
from ultralytics import YOLO

# 1. 加载模型
model = YOLO('yolov8n.pt')

# 2. 导出 INT8 ONNX（opset=13）
exported_path = model.export(
    format='onnx',
    int8=True,
    data='SplitDataset/data.yaml',
    opset=13
)

# 3. 定义规范化的目标文件名
target_name = "yolov8n_opset13_int8.onnx"

# 4. 自动重命名
if os.path.exists(exported_path):
    os.rename(exported_path, target_name)
    print(f"✅ 文件已成功导出并规范重命名为: {target_name}")