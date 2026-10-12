import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO
import onnxruntime
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
    CalibrationMethod
)

# 1. 导出 FP32 ONNX 模型
print("=== 1. 导出 FP32 ONNX 模型 ===")
model = YOLO('yolov8n.pt')
onnx_fp32_path = model.export(format='onnx', opset=13)

# 2. 构建数据读取器
class YOLODataReader(CalibrationDataReader):
    def __init__(self, img_dir, img_size=(640, 640)):
        self.img_paths = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
        self.img_size = img_size
        self.enum_data = None
        print(f"📸 找到校准图片: {len(self.img_paths)} 张")

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(self.img_paths)
        
        path = next(self.enum_data, None)
        if path is None:
            return None

        img = cv2.imread(path)
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, axis=0)  # NCHW

        return {'images': img}

# 3. 对称 INT8 量化 (关键：禁用 Bias 量化)
print("=== 2. 开始对称 INT8 量化 (适配 Jetson TRT 8.5.2) ===")
data_reader = YOLODataReader('SplitDataset/images/val')
onnx_int8_output = 'yolov8n_symmetric_clean.onnx'

quantize_static(
    model_input=onnx_fp32_path,
    model_output=onnx_int8_output,
    calibration_data_reader=data_reader,
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    calibrate_method=CalibrationMethod.Entropy,
    extra_options={
        'ActivationSymmetric': True,  # 激活值对称量化 (zeroPoint = 0)
        'WeightSymmetric': True,      # 权重大体对称量化 (zeroPoint = 0)
        'QuantizeBias': False,        # 核心修复：禁止量化 Bias，防止产生 bias_DequantizeLinear 节点
    }
)

print(f"\n🎉 成功生成标准对称 INT8 ONNX 模型: {onnx_int8_output}")