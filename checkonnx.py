import onnx

# 加载并检查模型结构
onnx_model = onnx.load("yolov8n.onnx")
onnx.checker.check_model(onnx_model)
print("✅ ONNX 模型校验通过，文件完整无误！")