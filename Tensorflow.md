

## tensorflow指定使用的GPU

### 在终端执行程序时指定GPU

> CUDA_VISIBLE_DEVICES=1   python  your_file.py
>
> 这样在跑你的网络之前，告诉程序只能看到1号GPU，其他的GPU它不可见
>
> 可用的形式如下：
>
> CUDA_VISIBLE_DEVICES=1           Only device 1 will be seen
> CUDA_VISIBLE_DEVICES=0,1         Devices 0 and 1 will be visible
> CUDA_VISIBLE_DEVICES="0,1"       Same as above, quotation marks are optional
> CUDA_VISIBLE_DEVICES=0,2,3       Devices 0, 2, 3 will be visible; device 1 is masked
> CUDA_VISIBLE_DEVICES=""          No GPU will be visible



### 在Python代码中指定GPU

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

### 设置定量的GPU使用量

```python
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU90%的显存
session = tf.Session(config=config)
```

### **设置最小的GPU使用量**

```python
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
```

