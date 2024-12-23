## 多GPU并行与显存管理

[【Pytorch】多GPU并行与显存管理多gpu并行 ](https://blog.csdn.net/ccamelliatree/article/details/106299615)





## pytorch强化学习





## pt导出模型



```python
import torch

def save_trace_deploy(model, save_path="DSNTt.pt"):
    model.eval()
    input = torch.randn(1, 1, 128, 128, 200, device=device)
    traced_trace_model = torch.jit.trace(model, input)
    traced_trace_model.save(save_path)


def save_script_deploy(model, save_path="DSNTs.pt"):
    model.eval()
    # Step 1 : Converting to Torch Script via anntotation
    traced_script_module = torch.jit.script(model)
    # Step 2: Serializing the Script Module to a File
    traced_script_module.save(save_path)
```



## 基于pt模型的python推理

