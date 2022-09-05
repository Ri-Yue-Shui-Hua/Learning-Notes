# 神经网络笔记



## 3DUNet



支持任意尺寸的输入,关键代码：

```cpp
 def center_crop(self, skip_connection, x):
        skip_shape = torch.tensor(skip_connection.shape)
        x_shape = torch.tensor(x.shape)
        crop = skip_shape[2:] - x_shape[2:]
        half_crop = (crop / 2).int()
        # If skip_connection is 10, 20, 30 and x is (6, 14, 12)
        # Then pad will be (-2, -2, -3, -3, -9, -9)
        pad = -torch.stack((half_crop, half_crop)).t().flatten()
        skip_connection = F.pad(skip_connection, pad.tolist())
        return skip_connection
```

解码器中调用：

```cpp
 def forward(self, skip_connection, x):
        x = self.upsample(x)
        skip_connection = self.center_crop(skip_connection, x)
        x = torch.cat((skip_connection, x), dim=CHANNELS_DIMENSION)
        if self.residual:
            connection = self.conv_residual(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x += connection
        else:
            x = self.conv1(x)
            x = self.conv2(x)
        return x
```



## 神经网络基础

FCN为什么需要固定尺寸的输入？

一个确定的CNN网络结构之所以要固定输入图片大小，是因为全连接层权值数固定，而该权值数和feature map大小有关, 但是FCN在CNN的基础上把1000个结点的全连接层改为含有1000个1×1卷积核的卷积层，经过这一层，还是得到二维的feature map，同样我们也不关心这个feature map大小, 所以对于输入图片的size并没有限制。

# 深度学习常见问题

## 深度学习常用方法

现在在应用领域应用的做多的是DNN，CNN和RNN。

- DNN是传统的全连接网络，可以用于广告点击率预估，推荐等。其使用embedding的方式将很多离散的特征编码到神经网络中，可以很大的提升结果。

- CNN主要用于计算机视觉(Computer Vision)领域，CNN的出现主要解决了DNN在图像领域中参数过多的问题。同时，CNN特有的卷积、池化、batch normalization、Inception、ResNet、DeepNet等一系列的发展也使得在分类、物体检测、人脸识别、图像分割等众多领域有了长足的进步。同时，CNN不仅在图像上应用很多，在自然语言处理上也颇有进展，现在已经有基于CNN的语言模型能够达到比LSTM更好的效果。在最新的AlphaZero中，CNN中的ResNet也是两种基本算法之一。

- GAN是一种应用在生成模型的训练方法，现在有很多在CV方面的应用，例如图像翻译，图像超清化、图像修复等等。

## 简单说说CNN常用的几个模型

![image-20220811213507832](DeepLearning.assets/image-20220811213507832.png)



## 在神经网络中，有哪些办法防止过拟合

1. Dropout

2. 加L1/L2正则化

3. BatchNormalization

4. 网络bagging

5. 提取终止训练

6. 数据增强



## CNN是什么，CNN关键的层有哪些？

其关键层有：

① 输入层，对数据去均值，做data augmentation等工作

② 卷积层，局部关联抽取feature

③ 激活层，非线性变化

④ 池化层，下采样

⑤ 全连接层，增加模型非线性

⑥ 高速通道，快速连接

⑦ BN层，缓解梯度弥散

## 如何解决深度学习中模型训练效果不佳的情况？

如果模型的训练效果不好，可先考察以下几个方面是否有可以优化的地方。

(1)选择合适的损失函数（choosing proper loss ）

神经网络的损失函数是非凸的，有多个局部最低点，目标是找到一个可用的最低点。非凸函数是凹凸不平的，但是不同的损失函数凹凸起伏的程度不同，例如下述的平方损失和交叉熵损失，后者起伏更大，且后者更容易找到一个可用的最低点，从而达到优化的目的。

\- Square Error（平方损失）

\- Cross Entropy（交叉熵损失）

(2)选择合适的Mini-batch size

采用合适的Mini-batch进行学习，使用Mini-batch的方法进行学习，一方面可以减少计算量，一方面有助于跳出局部最优点。因此要使用Mini-batch。更进一步，batch的选择非常重要，batch取太大会陷入局部最小值，batch取太小会抖动厉害，因此要选择一个合适的batch size。



# 深度学习实践



## pytorch模型导出onnx模型



```python
def export_onnx(model, input, input_names, output_names, modelname):
    dummy_input = input
    torch.onnx.export(model, dummy_input, modelname,
                      export_params=True,
                      verbose=False,
                      opset_version=13,
                      input_names=input_names,
                      output_names=output_names, dynamic_axes={'input': {2: "input_dynamic_axes_1", 3: "input_dynamic_axes_1",
                    4: "input_dynamic_axes_1"}, 'output': {2: "input_dynamic_axes_1", 3: "input_dynamic_axes_1", 4: "input_dynamic_axes_1"}})
    # torch.onnx.export(model, dummy_input, modelname,
    #                   export_params=True,
    #                   verbose=False,
    #                   opset_version=12,
    #                   input_names=input_names,
    #                   output_names=output_names)
    print("export onnx model success!")
    
 with torch.no_grad():
        # net.eval()
        input = torch.randn(1, 1, 224, 144, 144, device=device)
        # itk_img = sitk.ReadImage('D:/Dataset/Hip_Femur_Detection/Detect/BJ_02006.nii.gz')
        # img_arr = sitk.GetArrayFromImage(itk_img)
        # img_arr = img_arr[np.newaxis, np.newaxis, :, :, :]
        # input = torch.from_numpy(img_arr).float()
        input_names = ['input']
        output_names = ['output']
        export_onnx(net, input, input_names, output_names, "test.onnx")

        exit(0)
```

## 反卷积输出尺寸计算

[ConvTranspose3d — PyTorch 1.12 documentation](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose3d.html?highlight=convtranspose3d#torch.nn.ConvTranspose3d)





##  根据网络结构图反向工程网络

![test.onnx](DeepLearning.assets/test.onnx.svg)



```python
# -*- coding : UTF-8 -*-
# @file   : UNET_3D.py
# @Time   : 2022/8/11 0011 21:50
# @Author : wmz

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels=[32, 64],
                 kernel_size=3, padding=1):
        super(DoubleConv, self).__init__()
        if isinstance(out_channels, int):
            out_channels = [out_channels] * 2

        self.doubleConv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels[0],
                      kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels[0], out_channels[1],
                      kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels[1]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.doubleConv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels=[32, 64]):
        super(Down, self).__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        # self.trans_conv = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.trans_conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, dilation=2, padding=1, output_padding=1)

        self.double_conv = DoubleConv(out_channels + out_channels, out_channels)

    def forward(self, x_trans, x_cat):
        x = self.trans_conv(x_trans)
        #print("x_trans", x_trans.shape)
        #print("x", x.shape)
        #print("x_cat", x_cat.shape)
        x = torch.cat([x_cat, x], dim=1)
        return self.double_conv(x)


class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Out, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        #self.bn = nn.BatchNorm3d(out_channels)
        self.norm = nn.ReLU()
        #self.norm = nn.Tanh()

    def forward(self, x):
        return self.norm(self.conv(x))


ch = [32,32,64,128,256,512]
class UNet_3D(nn.Module):
    def __init__(self, in_channels, out_classes, last_layer=Out):
        super(UNet_3D, self).__init__()
        self.encoder_1 = DoubleConv(in_channels, [ch[0], ch[1]]) # 32,32
        self.encoder_2 = Down(ch[1], [ch[2], ch[2]]) #32, 64 64
        self.encoder_3 = Down(ch[2], [ch[3], ch[3]]) # 64,128,128
        self.encoder_4 = Down(ch[3], [ch[4], ch[4]])  # 128,256,256
        self.buttom = Down(ch[4], [ch[5], ch[5]]) # 256,512,512
        self.decoder_4 = Up(ch[5], ch[4])  # 512,256
        self.decoder_3 = Up(ch[4], ch[3]) #256,128
        self.decoder_2 = Up(ch[3], ch[2]) # 128, 64
        self.decoder_1 = Up(ch[2], ch[1]) # 64, 32
        self.output = last_layer(ch[1], out_classes) if last_layer is not None else None

    def forward(self, x):
        e1 = self.encoder_1(x)
        e2 = self.encoder_2(e1)
        e3 = self.encoder_3(e2)
        e4 = self.encoder_4(e3)
        e = self.buttom(e4)
        #print(e.shape, e3.shape)
        d4 = self.decoder_4(e, e4)
        d3 = self.decoder_3(d4, e3)
        d2 = self.decoder_2(d3, e2)
        d1 = self.decoder_1(d2, e1)
        if self.output is not None:
            d1 = self.output(d1)
        return d1


def export_onnx(model, input, input_names, output_names, modelname):
    dummy_input = input
    torch.onnx.export(model, dummy_input, modelname,
                      export_params=True,
                      verbose=False,
                      opset_version=13,
                      input_names=input_names,
                      output_names=output_names, dynamic_axes={'input': {2: "input_dynamic_axes_1", 3: "input_dynamic_axes_1",
                    4: "input_dynamic_axes_1"}, 'output': {2: "input_dynamic_axes_1", 3: "input_dynamic_axes_1", 4: "input_dynamic_axes_1"}})
    # torch.onnx.export(model, dummy_input, modelname,
    #                   export_params=True,
    #                   verbose=False,
    #                   opset_version=12,
    #                   input_names=input_names,
    #                   output_names=output_names)
    print("export onnx model success!")


if __name__ == "__main__":
    import numpy as np
    device = torch.device('cpu')

    x = np.random.rand(1, 1, 224, 144, 144)
    pt = torch.from_numpy(x).float()
    net = UNet_3D(in_channels=1, out_classes=1)
    # net.train()
    y = net(pt)
    with torch.no_grad():
        # net.eval()
        input = torch.randn(1, 1, 224, 144, 144, device=device)
        # itk_img = sitk.ReadImage('D:/Dataset/Hip_Femur_Detection/Detect/BJ_02006.nii.gz')
        # img_arr = sitk.GetArrayFromImage(itk_img)
        # img_arr = img_arr[np.newaxis, np.newaxis, :, :, :]
        # input = torch.from_numpy(img_arr).float()
        input_names = ['input']
        output_names = ['output']
        export_onnx(net, input, input_names, output_names, "test.onnx")

        exit(0)
    print(y.shape)



```



## 神经网络数据加载

[torch.utils.data — PyTorch 1.12 documentation](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset)



## onnx模型

[[ONNX从入门到放弃\] 1. ONNX协议基础 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/456124635)

 ONNX是开放式[神经网络](https://so.csdn.net/so/search?q=神经网络&spm=1001.2101.3001.7020)(Open Neural Network Exchange)的简称，主要由微软和合作伙伴社区创建和维护。

###  onnx结构

通过阅读github上ONNX工程中的.proto文件，很容易发现ONNX由下面几部分组成。

| 类型           | 用途                                                         |
| -------------- | ------------------------------------------------------------ |
| ModelProto     | 定义了整个网络的模型结构                                     |
| GraphProto     | 定义了模型的计算逻辑，包含了构成图的节点，这些节点组成了一个有向图结构 |
| NodeProto      | 定义了每个OP的具体操作                                       |
| ValueInfoProto | 序列化的张量，用来保存weight和bias                           |
| TensorProto    | 定义了输入输出形状信息和张量的维度信息                       |
| AttributeProto | 定义了OP中的具体参数，比如Conv中的stride和kernel_size等      |

在工程实践中，导出一个ONNX模型就是导出一个ModelProto。ModelProto中包含GraphProto、版本号等信息。GraphProto又包含了NodeProto类型的node，ValueInfoProto类型的input和output，TensorProto类型的initializer。其中，node包含了模型中的所有OP，input和output包含了模型的输入和输出，initializer包含了所有的权重信息。 每个计算节点node中还包含了一个AttributeProto数组，用来描述该节点的属性，比如Conv节点或者卷积层的属性包含group，pad，strides等等。

### 构建ONNX模型



在熟悉了ONNX的模型结构之后，可以采用ONNX官方提供的API构建ONNX的网络结构，这里举一个简单的例子。

```python
# -*- coding : UTF-8 -*-
# @file   : test_onnx.py
# @Time   : 2022-09-03 18:24
# @Author : wmz

import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

if __name__ == '__main__':
    # 创建输入（ValueInfoProto）
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3, 2])
    pads = helper.make_tensor_value_info('pads', TensorProto.FLOAT, [1, 4])
    value = helper.make_tensor_value_info('value', AttributeProto.FLOAT, [1])

    # 创建输出 （ValueInfoProto）
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 4])

    # 创建节点（NodeProto）
    node_def = helper.make_node(
        'Pad',  # node name
        ['X', 'pads', 'value'],  # inputs
        ['Y'],  # outputs
        mode='constant',  # attributes
    )

    # 创建图 （GraphProto）
    graph_def = helper.make_graph(
        [node_def],
        'test_model',
        [X, pads, value],
        [Y],
    )

    # 创建模型(ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    print('The model is: \n{}'.format(model_def))
    onnx.checker.check_model(model_def)
    onnx.save(model_def, "./test_model.onnx")
    print('The model is checked!')

```

该例从onnx.helper模块中引入了多个函数，helper模块相当于对onnx的protobuf输出做了简单的封装，用户可以直接调用helper模块中提供的接口构造ONNX模型。按照之前提到过的ONNX结构定义，先构建模型的输入与输出，再构建模型的多个节点操作，用图将节点封装，一个复杂的模型中可能包含多个子图，所以图是模型的子集，最后再定义模型本身，构建相应的参数，打印的结果如下所示：

```bash
The model is: 
ir_version: 8
producer_name: "onnx-example"
graph {
  node {
    input: "X"
    input: "pads"
    input: "value"
    output: "Y"
    op_type: "Pad"
    attribute {
      name: "mode"
      s: "constant"
      type: STRING
    }
  }
  name: "test_model"
  input {
    name: "X"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 3
          }
          dim {
            dim_value: 2
          }
        }
      }
    }
  }
  input {
    name: "pads"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 1
          }
          dim {
            dim_value: 4
          }
        }
      }
    }
  }
  input {
    name: "value"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 1
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 3
          }
          dim {
            dim_value: 4
          }
        }
      }
    }
  }
}
opset_import {
  version: 15
}

The model is checked!

Process finished with exit code 0

```

### AI模型可视化

生成模型后，往往需要对导出的模型结构进行查看，这时模型可视化就显得十分重要。

目前，github上已经有多款开源的AI模型可视化工具，它们都有自己的特色，如TensorBoard，Netscope等。这里介绍一个非常好用的可视化工具Netron。

用Netron打开用onnx.helper导出的模型：

![image-20220903185854367](DeepLearning.assets/image-20220903185854367.png)



[(2条消息) ONNX构建并运行模型_亦梦云烟的博客-CSDN博客_onnx模型](https://blog.csdn.net/u010580016/article/details/119493797)

### ONNX文件格式

ONNX文件是基于Protobuf进行序列化。了解Protobuf协议的同学应该知道，Protobuf都会有一个*.proto的文件定义协议，ONNX的该协议定义在https://github.com/onnx/onnx/blob/master/onnx/onnx.proto3 文件中。

从onnx.proto3协议中我们需要重点知道的数据结构如下：

- ModelProto：模型的定义，包含版本信息，生产者和GraphProto。
- GraphProto: 包含很多重复的NodeProto, initializer, ValueInfoProto等，这些元素共同构成一个计算图，在GraphProto中，这些元素都是以列表的方式存储，连接关系是通过Node之间的输入输出进行表达的。
- NodeProto: onnx的计算图是一个有向无环图(DAG)，NodeProto定义算子类型，节点的输入输出，还包含属性。
- ValueInforProto: 定义输入输出这类变量的类型。
- TensorProto: 序列化的权重数据，包含数据的数据类型，shape等。
- AttributeProto: 具有名字的属性，可以存储基本的数据类型(int, float, string, vector等)也可以存储onnx定义的数据结构(TENSOR, GRAPH等)。



### 搭建ONNX模型
 ONNX是用DAG来描述网络结构的，也就是一个网络(Graph)由节点(Node)和边(Tensor)组成，ONNX提供的helper类中有很多API可以用来构建一个ONNX网络模型，比如make_node, make_graph, make_tensor等，下面是一个单个Conv2d的网络构造示例：

```python
# -*- coding : UTF-8 -*-
# @file   : onnx_p2.py
# @Time   : 2022-09-03 19:10
# @Author : wmz
import onnx
from onnx import helper
from onnx import TensorProto
import numpy as np

if __name__ == '__main__':
    weight = np.random.randn(36)
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2, 4, 4])
    W = helper.make_tensor('W', TensorProto.FLOAT, [2, 2, 3, 3], weight)
    B = helper.make_tensor('B', TensorProto.FLOAT, [2], [1.0, 2.0])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2, 2, 2])

    node_def = helper.make_node(
        'Conv',  # node name
        ['X', 'W', 'B'],
        ['Y'],  # outputs
        # attributes
        strides=[2, 2],
    )

    graph_def = helper.make_graph(
        [node_def],
        'test_conv_mode',
        [X],  # graph inputs
        [Y],  # graph outputs
        initializer=[W, B],
    )

    mode_def = helper.make_model(graph_def, producer_name='onnx-example')
    onnx.checker.check_model(mode_def)
    onnx.save(mode_def, "./Conv.onnx")

```



搭建的这个Conv[算子](https://so.csdn.net/so/search?q=算子&spm=1001.2101.3001.7020)模型使用netron可视化如下图所示：

![image-20220903191300852](DeepLearning.assets/image-20220903191300852.png)



这个示例演示了如何使用helper的make_tensor_value_info, make_mode, make_graph, make_model等方法来搭建一个onnx模型。

        相比于PyTorch或其它框架，这些API看起来仍然显得比较繁琐，一般我们也不会用ONNX来搭建一个大型的网络模型，而是通过其它框架转换得到一个ONNX模型。


### Shape Inference

很多时候我们从pytorch, [tensorflow](https://so.csdn.net/so/search?q=tensorflow&spm=1001.2101.3001.7020)或其他框架转换过来的onnx模型中间节点并没有shape信息，如下图所示：

![image-20220903191654776](DeepLearning.assets/image-20220903191654776.png)



 我们经常希望能直接看到网络中某些node的shape信息，shape_inference模块可以推导出所有node的shape信息，这样可视化模型时将会更友好：



```python
import onnx
from onnx import shape_inference
 
onnx_model = onnx.load("./test_data/mobilenetv2-1.0.onnx")
onnx_model = shape_inference.infer_shapes(onnx_model)
onnx.save(onnx_model, "./test_data/mobilenetv2-1.0_shaped.onnx")
```

可视化经过shape_inference之后的模型如下图：

![image-20220903191757681](DeepLearning.assets/image-20220903191757681.png)





### ONNX Optimizer

ONNX的optimizer模块提供部分图优化的功能，例如最常用的：fuse_bn_into_conv，fuse_pad_into_conv等等。

​    查看onnx支持的优化方法：

```python
from onnx import optimizer
all_passes = optimizer.get_available_passes()
print("Available optimization passes:")
for p in all_passes:
    print(p)
print()
```



[(2条消息) 模型部署入门教程（五）：ONNX 模型的修改与调试_OpenMMLab的博客-CSDN博客_onnx模型](https://blog.csdn.net/qq_39967751/article/details/124989296)



## onnx模型加密



[(2条消息) onnxruntime(c++)模型加密与解密部署_小小菜鸡升级ing的博客-CSDN博客_onnx模型加密](https://blog.csdn.net/weixin_39853245/article/details/117953268)



```cpp
//加密模型
#ifdef WINVER
void encryptDecrypt(const wchar_t* toEncrypt, int strLength, const wchar_t* key, wchar_t* output)
{
        int keyLength = wcslen(key);
        for (int i = 0; i < strLength; i++)
        {
                output[i] = toEncrypt[i] ^ key[i % keyLength];
        }
}
#else
void encryptDecrypt(const char* toEncrypt, int strLength, const char* key, wchar_t* output)
{
        int keyLength = strlen(key);
        for (int i = 0; i < strLength; i++)
        {
                output[i] = toEncrypt[i] ^ key[i % keyLength];
        }
}
#endif



int main()
{
        bool bRet = true;
        std::string model_path = "11.onnx";
        FILE *pModel1File = fopen(model_path.c_str(), "rb");
        if (NULL == pModel1File)
        {
                bRet = false;

        }
        int length = 0;
        fseek(pModel1File, 0, SEEK_END);
        length = ftell(pModel1File);
        fseek(pModel1File, 0, SEEK_SET);
        ORTCHAR_T* model1rbuffer = nullptr;
        model1rbuffer = new ORTCHAR_T[length];
        if (nullptr == model1rbuffer) {
                bRet = false;
        }

        fread((ORTCHAR_T *)model1rbuffer, 1, length, pModel1File);
        fclose(pModel1File);

        //写入加密模型
        FILE *pOutFile = fopen("1.data", "wb");
        if (NULL == pOutFile)
        {
                return -1;
        }
        #ifdef WINVER
        const wchar_t* key = L"1234";
        const wchar_t* encrypted = new wchar_t[length];
        #else
        const char* key = "1234";
        const char* encrypted = new char[length];
        #endif
        encryptDecrypt(model1rbuffer, length, key, encrypted);
        fwrite(encrypted, 1,  length, pOutFile);
        fclose(pOutFile);
        delete[] encrypted;
        delete[] model1rbuffer;
        system("pause");
        return 0;
}


// 解密模型加载
int main() 
{
        bool bRet = true;
        std::string model_path = "1.data";
        FILE *pModel1File = fopen(model_path.c_str(), "rb");
        if (NULL == pModel1File)
        {
                bRet = false;
        }
        int length = 0;
        fseek(pModel1File, 0, SEEK_END);
        length = ftell(pModel1File);
        fseek(pModel1File, 0, SEEK_SET);
        ORTCHAR_T* model1rbuffer = nullptr;
        model1rbuffer = new ORTCHAR_T[length];
        if (nullptr == model1rbuffer) {
                bRet = false;
        }

        fread((ORTCHAR_T *)model1rbuffer, 1, length, pModel1File);
        fclose(pModel1File);
        #ifdef WINVER
        	const wchar_t* key = L"1234";
        	const wchar_t* encrypted = new wchar_t[length];
        #else
        	const char* key = "1234";
        	const char* encrypted = new char[length];
        #endif
        
        encryptDecrypt(model1rbuffer, length, key, decrypted);

        const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        OrtEnv* env_;
        g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "cyril", &env_);
        OrtSessionOptions* session_options_= nullptr;
        OrtSession* session_= nullptr;
        g_ort->CreateSessionOptions(&session_options_);
        g_ort->SetIntraOpNumThreads(session_options_, 1);
        g_ort->SetSessionGraphOptimizationLevel(session_options_, ORT_ENABLE_ALL);
        //g_ort->CreateSession(env_, model1rbuffer, session_options_, &session_);
        g_ort->CreateSessionFromArray(env_, decrypted, length, session_options_, &session_);
        std::cout << "ok!" << std::endl;
        delete[] model1rbuffer;
        delete[] decrypted;
        system("pause");
        return 0;
}


```





































