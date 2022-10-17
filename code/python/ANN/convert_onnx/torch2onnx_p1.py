import torch
from torch import nn, Tensor

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image: Tensor) -> Tensor:
        outputs = torch.softmax(image, 1)
        return outputs


if __name__ == "__main__":

    batch_size = 1  #批处理大小
    input_shape = (2, 2)   #输入数据,改成自己的输入shape
    # #set the model to inference mode
    model = Model()
    model.eval()
    x = torch.randn(batch_size, *input_shape)	# 生成张量
    export_onnx_file = "test.onnx"			# 目的ONNX文件名
    torch.onnx.export(model,
                    x,
                    export_onnx_file,
                    opset_version=10,
                    do_constant_folding=True,	# 是否执行常量折叠优化
                    input_names=["input"],	# 输入名
                    output_names=["output"],	# 输出名
                    dynamic_axes={"input":{0:"batch_size"},  # 批处理变量
                                    "output":{0:"batch_size"}})