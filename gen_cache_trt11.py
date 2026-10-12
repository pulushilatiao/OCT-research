import os
import glob
import cv2
import numpy as np
import torch
import tensorrt as trt
from ultralytics import YOLO

print("=== 1. 导出标准的 FP32 ONNX 模型 ===")
model = YOLO('yolov8n.pt')
onnx_name = 'FP32-INT8_yolov8n.onnx'
onnx_raw_path = model.export(format='onnx', opset=13)
if os.path.exists(onnx_raw_path):
    os.rename(onnx_raw_path, onnx_name)

# 动态兼容 TRT 10/11 和旧版 TRT 8/9 的校准器类名
if hasattr(trt, 'IInt8EntropyCalibrator2'):
    BaseCalibrator = trt.IInt8EntropyCalibrator2
else:
    BaseCalibrator = trt.IInt8Calibrator

class YOLOEntropyCalibrator(BaseCalibrator):
    def __init__(self, img_dir, cache_file, img_size=(640, 640)):
        super().__init__()
        self.cache_file = cache_file
        self.img_size = img_size
        self.img_paths = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
        self.batch_size = 1
        self.current_idx = 0
        self.device_input = torch.empty((1, 3, img_size[0], img_size[1]), dtype=torch.float32, device='cuda')
        print(f"📸 找到校准图片数量: {len(self.img_paths)} 张")

    def get_algorithm(self):
        # 针对 TRT 10/11 必须显式指定校准算法
        return trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_idx >= len(self.img_paths):
            return None
        
        path = self.img_paths[self.current_idx]
        self.current_idx += 1
        
        # BGR -> RGB, Resize, Normalize 0-1
        img = cv2.imread(path)
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1) 
        
        tensor = torch.from_numpy(img).unsqueeze(0).cuda()
        self.device_input.copy_(tensor)
        return [int(self.device_input.data_ptr())]

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

print("=== 2. 开始运行 TensorRT 原生 INT8 校准 ===")
logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open(onnx_name, 'rb') as f:
    if not parser.parse(f.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        raise RuntimeError("ONNX 解析失败")

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.INT8)

# 传入数据集进行校准
cache_filename = 'yolov8n.cache'
config.int8_calibrator = YOLOEntropyCalibrator(img_dir='SplitDataset/images/val', cache_file=cache_filename)

print("正在计算校准因子 (可能需要1-3分钟)...")
_ = builder.build_serialized_network(network, config)

if os.path.exists(cache_filename):
    print(f"\n🎉 成功在当前目录生成 TensorRT 校准表: {cache_filename}")
else:
    print("\n❌ 未能生成校准表。")