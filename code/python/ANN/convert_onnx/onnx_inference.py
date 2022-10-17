import onnxruntime as ort
import onnx
import numpy as np

if __name__ == "__main__":
    model = onnx.load("test.onnx") # 加载onnx
    onnx.checker.check_model(model) # 检查生成模型是否错误
    session = ort.InferenceSession("test.onnx")
    input_shape = (2, 2)
    x = np.random.randn(1, *input_shape)
    print(x)

    inputs = {"input":x.astype(np.float32)}
    output = session.run(["output"], inputs)
    print(output)