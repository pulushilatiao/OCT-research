
from ultralytics import YOLO

# 1. 加载模型
model = YOLO('yolov8n.pt')

# 2. 导出为 INT8 Engine (PC 端会自动完成图片校准，并生成 INT8 权重与校准文件)
model.export(
    format='engine', int8=True, data='SplitDataset/data.yaml', batch=1, device=0
)