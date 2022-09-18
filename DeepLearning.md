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









# DL 各种机制和模块



## SCE模块

[CV中的attention机制之（cSE，sSE，scSE）_just-solo的博客-CSDN博客_scse注意力](https://blog.csdn.net/justsolow/article/details/106517945)



[最简单最易实现的SE模块（Squeeze-and-Excitation Networks）_just-solo的博客-CSDN博客_神经网络se模块](https://blog.csdn.net/justsolow/article/details/105376899)



Squeeze-and-Excitation Networks

SENet是Squeeze-and-Excitation Networks的简称，拿到了ImageNet2017分类比赛冠军，其效果得到了认可，其提出的SE模块思想简单，易于实现，并且很容易可以加载到现有的网络模型框架中。SENet主要是学习了channel之间的相关性，筛选出了针对通道的注意力，稍微增加了一点计算量，但是效果比较好。



![image-20220907205555633](DeepLearning.assets/image-20220907205555633.png)



> 本文介绍了一个用于语义分割领域的attention模块scSE。scSE模块与之前介绍的BAM模块很类似，不过在这里scSE模块只在语义分割中进行应用和测试，对语义分割准确率带来的提升比较大，还可以让分割边界更加平滑。

提出scSE模块论文的全称是：《**Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks** 》。这篇文章对SE模块进行了改进，提出了SE模块的三个变体cSE、sSE、scSE，并通过实验证明了了这样的模块可以增强有意义的特征，抑制无用特征。实验是基于两个医学上的数据集MALC Dataset和Visceral Dataset进行实验的。

语义分割模型大部分都是类似于U-Net这样的encoder-decoder的形式，先进行下采样，然后进行上采样到与原图一样的尺寸。其添加SE模块可以添加在每个卷积层之后，用于对feature map信息的提炼。具体方案如下图所示：

![image-20220907205933477](DeepLearning.assets/image-20220907205933477.png)





然后开始分别介绍由SE改进的三个模块，首先说明一下图例:

![image-20220907210001125](DeepLearning.assets/image-20220907210001125.png)

### cSE模块







![image-20220907210031029](DeepLearning.assets/image-20220907210031029.png)





这个模块类似之前BAM模块里的Channel attention模块，通过观察这个图就很容易理解其实现方法，具体流程如下:

- 将feature map通过global average pooling方法从[C, H, W]变为[C, 1, 1]
- 然后使用两个1×1×1卷积进行信息的处理，最终得到C维的向量
- 然后使用sigmoid函数进行归一化，得到对应的mask
- 最后通过channel-wise相乘，得到经过信息校准过的feature map



```python
import torch
import torch.nn as nn


class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels,
                                      in_channels // 2,
                                      kernel_size=1,
                                      bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels // 2,
                                         in_channels,
                                         kernel_size=1,
                                         bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)  # shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z)  # shape: [bs, c/2, 1, 1]
        z = self.Conv_Excitation(z)  # shape: [bs, c, 1, 1]
        z = self.norm(z)
        return U * z.expand_as(U)


if __name__ == "__main__":
    bs, c, h, w = 10, 3, 64, 64
    in_tensor = torch.ones(bs, c, h, w)

    c_se = cSE(c)
    print("in shape:", in_tensor.shape)
    out_tensor = c_se(in_tensor)
    print("out shape:", out_tensor.shape)
```

输出：

```bash
in shape: torch.Size([10, 3, 64, 64])
out shape: torch.Size([10, 3, 64, 64])
```

### sSE模块：





![image-20220907210238990](DeepLearning.assets/image-20220907210238990.png)





上图是空间注意力机制的实现，与BAM中的实现确实有很大不同，实现过程变得很简单，具体分析如下：

- 直接对feature map使用1×1×1卷积, 从[C, H, W]变为[1, H, W]的features
- 然后使用sigmoid进行激活得到spatial attention map
- 然后直接施加到原始feature map中，完成空间的信息校准

NOTE: 这里需要注意一点，先使用1×1×1卷积，后使用sigmoid函数，这个信息无法从图中直接获取，需要理解论文。



```python
import torch
import torch.nn as nn


class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U) # U:[bs,c,h,w] to q:[bs,1,h,w]
        q = self.norm(q)
        return U * q # 广播机制


if __name__ == "__main__":
    bs, c, h, w = 10, 3, 64, 64
    in_tensor = torch.ones(bs, c, h, w)

    s_se = sSE(c)
    print("in shape:", in_tensor.shape)
    out_tensor = s_se(in_tensor)
    print("out shape:", out_tensor.shape)
```

输出：

```bash
in shape: torch.Size([10, 3, 64, 64])
out shape: torch.Size([10, 3, 64, 64])
```

### scSE模块：





![image-20220907212515313](DeepLearning.assets/image-20220907212515313.png)





可以看出scSE是前两个模块的并联，与BAM的并联很相似，具体就是在分别通过sSE和cSE模块后，然后将两个模块相加，得到更为精准校准的feature map, 直接上代码：



```python
import torch
import torch.nn as nn


class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)  # U:[bs,c,h,w] to q:[bs,1,h,w]
        q = self.norm(q)
        return U * q  # 广播机制

class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels//2, in_channels, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)# shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z) # shape: [bs, c/2]
        z = self.Conv_Excitation(z) # shape: [bs, c]
        z = self.norm(z)
        return U * z.expand_as(U)

class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return U_cse+U_sse

if __name__ == "__main__":
    bs, c, h, w = 10, 3, 64, 64
    in_tensor = torch.ones(bs, c, h, w)

    sc_se = scSE(c)
    print("in shape:",in_tensor.shape)
    out_tensor = sc_se(in_tensor)
    print("out shape:", out_tensor.shape)
```

输出：

```bash
in shape: torch.Size([10, 3, 64, 64])
out shape: torch.Size([10, 3, 64, 64])
```



## 3D SCE模块

 主要是对上面的2D的算子换成3D的算子。

```python
class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1x1 = nn.Conv3d(in_channels, 1, 1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.conv1x1(U)
        q = self.norm(q)
        return U * q  # 广播机制


class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conv_squeeze = nn.Conv3d(
            in_channels, in_channels//2, 1, bias=False)
        self.conv_excitation = nn.Conv3d(
            in_channels//2, in_channels, 1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.pool(U)
        z = self.conv_squeeze(z)
        z = self.conv_excitation(z)
        z = self.norm(z)
        return U * z.expand_as(U)


class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.sSE = sSE(in_channels)
        self.cSE = cSE(in_channels)

    def forward(self, U):
        U_sSE = self.sSE(U)
        U_cSE = self.cSE(U)
        return U_sSE + U_cSE

```





## DSNT模块

论文地址：[Numerical Coordinate Regression with Convolutional Neural Networks](https://arxiv.org/abs/1801.07372)
代码地址：https://github.com/anibali/dsntnn

比起主流的预测heatmap再使用argmax获得最大响应点的方法，作者提出用soft-argmax方法，直接从featuremap计算得到近似的最大响应点，从而模型可以直接对坐标值进行回归预测。

该方法的好处是：

- 可以极大节约显存，减少了很多代码量（省去了高斯热图的生成）;
- 计算速度很快（省去argmax）;
- 训练过程全部可微分;
- 能取得还不错的效果;

但是从近年的新论文实验对比，基于坐标回归的方法，先天缺少空间和上下文信息，这限制了方法的性能上限

- DSNT这种regression-based method对移动端非常友好，在移动设备上为了提高fps往往特征图会压到14x14甚至7x7，这个尺度下heatmap根本用不了;
- 另一个好处是可以比较方便地混合使用2d和3d训练数据，由于每一轴上的坐标都是通过将特征图求和压缩到一维再求期望得到的，因此可以很容易地用2d数据监督3d特征图（2d只比3d少算一次回归）

[DSNT与SCN](https://blog.csdn.net/juluwangriyue/article/details/122890209)

### 论文总结

本文提供了一种从图像中直接学习到坐标的一种思路。现在主流的方法都是基于高斯核处理的heatmap作为监督，但这种方法学习到的heatmap，在后处理得到坐标的过程中，存在量化误差（比如4倍下采样的heatmap，量化误差的期望是2）。

本文提出一种新的处理方法，称为DSNT，通过DSNT处理（没添加额外参数），直接对坐标进行监督。DSNT是对heatmap进行处理的，思路如下图所示。最后的处理过程，就是将heatmap通过softmax，得到一个基于heatmap的概率分布，再通过这个概率分布，与预设好的X，Y（坐标轴）进行点乘，得到坐标的期望值。监督损失也是建立在这个期望值上的。



![image-20220917224742424](DeepLearning.assets/image-20220917224742424.png)



与其他方法比较：

![image-20220917224822768](DeepLearning.assets/image-20220917224822768.png)





![image-20220917225110228](DeepLearning.assets/image-20220917225110228.png)

虽然文中的思想，主要说的是直接对坐标进行的回归，但实际上应用时，还是对heatmap做了约束的，而且权重还不算小。换个角度想，其实本文的实际操作，也可以认为，是对heatmap做了监督，然后添加了一个坐标的正则化因子。该正则化项的监督，可以有效减少heatmap转化成坐标的量化损失，与一些直接对heatmap做回归造成的损失误差与预期不符的问题。但是，这个heatmap项的损失也是精心挑选的，甚至不添加heatmap损失项，比不少heatmap损失计算方法的结果更好一些。

但是，对于那些在图像中不存在的关键点（比如半身），以及多人之类的问题，DSNT都不能直接进行解决。对于某些场景的应用，这是不可避免的问题。

个人见解：之所以DSNT能直接得到坐标，又能同时具有空间泛化能力，是在于两点：（1）其对heatmap进行了监督，监督对象为高斯分布，具有对称性；（2）其对坐标轴对象X,Y进行了精心设计，分别是1 ∗ n和n ∗ 1的单方向性，使其在两个坐标轴具有对称性。

![image-20220917225653488](DeepLearning.assets/image-20220917225653488.png)



上面是论文的片段，其中 $X \in [\frac{-(n-1)}{n}, \frac{n-1}{n}] \in (-1,1)$与 $Y \in [\frac{-(m-1)}{m}, \frac{m-1}{m}] \in (-1,1)$
官方实现：

```python
import torch
def normalized_linspace(length, dtype=None, device=None):
    """Generate a vector with values ranging from -1 to 1.
    Note that the values correspond to the "centre" of each cell, so
    -1 and 1 are always conceptually outside the bounds of the vector.
    For example, if length = 4, the following vector is generated:
    ```text
     [ -0.75, -0.25,  0.25,  0.75 ]
     ^              ^             ^
    -1              0             1
```
    Args:
        length: The length of the vector
    Returns:
        The generated vector
    """
    if isinstance(length, torch.Tensor):
        length = length.to(device, dtype)
    first = -(length - 1.0) / length
    return torch.arange(length, dtype=dtype, device=device) * (2.0 / length) + first
```

官方实现没有包含边界值，下面的实现包含了边界值，有没有包含边界值的区别是，结果中会不会出现边界
对于二维的情况：

```python
 B, C, Y, X = h.shape
 x = torch.linspace(-1, 1, X)
 y = torch.linspace(-1, 1, Y)
```
对于三维的情况：

```python
B, C, Z, Y, X = h.shape
 x = torch.linspace(-1, 1, X)
 y = torch.linspace(-1, 1, Y)
 z = torch.linspace(-1, 1, Z)
```



![image-20220917230414881](DeepLearning.assets/image-20220917230414881.png)



![image-20220917230443944](DeepLearning.assets/image-20220917230443944.png)

![image-20220917230511983](DeepLearning.assets/image-20220917230511983.png)



将 $\hat{Z}_{i,j}$ 理解成概率密度函数，$X$和$Y$就是变量取值范围，我们的目标就是求期望，也就是目标位置。

如果是连续变量，求$X$方向的期望就是对$X$方向求积分，如果是离散变量，就是概率密度与离散变量的乘积和。
对于期望值，也就是目标值，对于二维坐标分别用$E_x,E_y$表示，对于三位坐标分别用$E_x,E_y,E_z$表示。
二维计算：

```python
Ex = (Z_hat * x).sum(dim=(2, 3)).sum(dim=-1)
Ey = (Z_hat * y).sum(dim=(2, 4)).sum(dim=-1)
```

三维计算：
```python
Ex = (Z_hat * x).sum(dim=(2, 3)).sum(dim=-1)
Ey = (Z_hat * y).sum(dim=(2, 4)).sum(dim=-1)
Ez = (Z_hat * z).sum(dim=(3, 4)).sum(dim=-1)
```



### DSNT 代码



使用说明：[dsntnn/examples/basic_usage.md](https://github.com/anibali/dsntnn/blob/master/examples/basic_usage.md)

下面通过一个坐标点回归任务来学习dsnt的使用

下面首先创建一个简单的FCN模型，

```python
import torch 
torch.manual_seed(12345)
from torch import nn
import dsntnn
class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
        )
    def forward(self, x):
        return self.layers(x)
```

DSNT层可以很方便的扩展到任何的FCN网络后面，来处理坐标点回归任务，下面就是一个例子

```python
class CoordRegressionNetwork(nn.Module):
    def __init__(self, n_locations):
        super().__init__()
        self.fcn = FCN()
        self.hm_conv = nn.Conv2d(16, n_locations, kernel_size=1, bias=False)
 
    def forward(self, images):
        # 1. Run the images through our FCN
        fcn_out = self.fcn(images)
        # 2. Use a 1x1 conv to get one unnormalized heatmap per location
        unnormalized_heatmaps = self.hm_conv(fcn_out)
        # 3. Normalize the heatmaps
        heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps)
        # 4. Calculate the coordinates
        coords = dsntnn.dsnt(heatmaps)
 
        return coords, heatmaps
```
训练模型

为了训练模型下面创建了一个小浣熊单只眼睛的图像

```python
from torch import optim
import matplotlib.pyplot as plt
import scipy
import scipy.misc as sm
from skimage.transform import resize
 
image_size = [40, 40]
image = scipy.misc.face()[200:400, 600:800, :]
raccoon_face = resize(image, image_size)  # (40, 40, 3)
eye_x, eye_y = 24, 26
 
plt.imshow(raccoon_face)
plt.scatter([eye_x], [eye_y], color='red', marker='X')
plt.show()
```

![image-20220917231231241](DeepLearning.assets/image-20220917231231241.png)



由于DSNT输出的坐标范围是(-1, 1)，所以需要将target的坐标也归一化到这个范围

```python
import torch
raccoon_face_tensor = torch.from_numpy(raccoon_face).permute(2, 0, 1).float()  # torch.Size([3, 40, 40])
input_tensor = raccoon_face_tensor.div(255).unsqueeze(0)  # torch.Size([1, 3, 40, 40])
input_var = input_tensor.cuda()
 
eye_coords_tensor = torch.Tensor([[[eye_x, eye_y]]])  # shape = [1, 1, 2],value=[[[24., 26.]]]
target_tensor = (eye_coords_tensor * 2 + 1) / torch.Tensor(image_size) - 1  # shape = [1, 1, 2],value=[[[0.2250, 0.3250]]]
target_var = target_tensor.cuda()
 
print('Target: {:0.4f}, {:0.4f}'.format(*list(target_tensor.squeeze())))
```
运行输出：

```bash
Target: 0.2250, 0.3250
```
现在我们训练模型，让模型在小浣熊的眼睛点出过拟合，全部的代码如下

```python
import scipy
import scipy.misc as sm
from skimage.transform import resize
import matplotlib.pyplot as plt
import torch
torch.manual_seed(12345)
from torch import nn
import dsntnn
import torch.optim as optim
class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
        )
    def forward(self, x):
        return self.layers(x)
class CoordRegressionNetwork(nn.Module):
    def __init__(self, n_locations):
        super().__init__()
        self.fcn = FCN()
        self.hm_conv = nn.Conv2d(16, n_locations, kernel_size=1, bias=False)
    def forward(self, images):
        # 1. Run the images through our FCN
        fcn_out = self.fcn(images)
        # 2. Use a 1x1 conv to get one unnormalized heatmap per location
        unnormalized_heatmaps = self.hm_conv(fcn_out)
        # 3. Normalize the heatmaps
        heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps)
        # 4. Calculate the coordinates
        coords = dsntnn.dsnt(heatmaps)
        return coords, heatmaps
image_size = [40, 40]
# raccoon_face = sm.imresize(scipy.misc.face()[200:400, 600:800, :], image_size)      # (40, 40, 3)
image = scipy.misc.face()[200:400, 600:800, :]
raccoon_face = resize(image, image_size)
eye_x, eye_y = 24, 26
plt.imshow(raccoon_face)
plt.scatter([eye_x], [eye_y], color='red', marker='X')
plt.show()
raccoon_face_tensor = torch.from_numpy(raccoon_face).permute(2, 0, 1).float()   # torch.Size([3, 40, 40])
input_tensor = raccoon_face_tensor.div(255).unsqueeze(0)    # torch.Size([1, 3, 40, 40])
input_var = input_tensor.cuda()
eye_coords_tensor = torch.Tensor([[[eye_x, eye_y]]])    # # shape = [1, 1, 2],value=[[[24., 26.]]]
target_tensor = (eye_coords_tensor * 2 + 1) / torch.Tensor(image_size) - 1  # shape = [1, 1, 2],value=[[[0.2250, 0.3250]]]
target_var = target_tensor.cuda()
print('Target: {:0.4f}, {:0.4f}'.format(*list(target_tensor.squeeze())))
model = CoordRegressionNetwork(n_locations=1).cuda()    # n_locations=keypoint num=1
# coords, heatmaps = model(input_var)
# print('Initial prediction: {:0.4f}, {:0.4f}'.format(*list(coords[0, 0])))
# plt.imshow(heatmaps[0, 0].detach().cpu().numpy())
# plt.show()
optimizer = optim.RMSprop(model.parameters(), lr=2.5e-4)
for i in range(400):
    # Forward pass
    coords, heatmaps = model(input_var)
    # coords:shape=[1, 1, 2], value=[[[0.0323, 0.0566]]]; heatmaps:shape=[1, 1, 40, 40]
    # Per-location euclidean losses
    euc_losses = dsntnn.euclidean_losses(coords, target_var)
    # Per-location regularization losses
    reg_losses = dsntnn.js_reg_losses(heatmaps, target_var, sigma_t=1.0)
    # Combine losses into an overall loss
    loss = dsntnn.average_loss(euc_losses + reg_losses)
    # Calculate gradients
    optimizer.zero_grad()
    loss.backward()
    # Update model parameters with RMSprop
    optimizer.step()
# Predictions after training
print('Predicted coords: {:0.4f}, {:0.4f}'.format(*list(coords[0, 0])))
plt.imshow(heatmaps[0, 0].detach().cpu().numpy())
plt.show()
```



运行输出：

![image-20220918100148843](DeepLearning.assets/image-20220918100148843.png)



Target: 0.2250, 0.3250 

Predicted coords: 0.2236, 0.3299

![image-20220918100218157](DeepLearning.assets/image-20220918100218157.png)





### DSNT的其他实现



```python
def DSNT_f(h, spacial=None):
    B, C, Z, Y, X = h.shape
    #heatmap = heatmap * 20
    #h = heatmap / torch.sum(heatmap, dim=(2, 3, 4), keepdim=True)
    x = torch.linspace(-1, 1, X).to(h.device)
    y = torch.linspace(-1, 1, Y).to(h.device)
    z = torch.linspace(-1, 1, Z).to(h.device)
    x_cord = x.view([B, 1, 1, 1, X])
    y_cord = y.view([B, 1, 1, Y, 1])
    z_cord = z.view([B, 1, Z, 1, 1])
    px = (h * x_cord).sum(dim=(2, 3)).sum(dim=-1)
    py = (h * y_cord).sum(dim=(2, 4)).sum(dim=-1)
    pz = (h * z_cord).sum(dim=(3, 4)).sum(dim=-1)

    #print(x_cord.shape, px.shape, px.view(B, C, 1, 1, 1).shape)
    var_x = (h * ((x_cord - px.view(B, C, 1, 1, 1)) ** 2)).sum(dim=(2, 3, 4))
    var_y = (h * (y_cord - py.view(B, C, 1, 1, 1)) ** 2).sum(dim=(2, 3, 4))
    var_z = (h * (z_cord - pz.view(B, C, 1, 1, 1)) ** 2).sum(dim=(2, 3, 4))
    return px, py, pz, var_x, var_y, var_z
```









## SCN模块



[1908.00748.pdf (arxiv.org)](https://arxiv.org/pdf/1908.00748.pdf)

论文图片

![image-20220917093008844](DeepLearning.assets/image-20220917093008844.png)



![image-20220917102836972](DeepLearning.assets/image-20220917102836972.png)

该方法的优点：

不需要大量的标注数据就可以训练较好的定位模型。

### SpatialConfiguration-Net

The fundamental concept of the SpatialConfiguration-Net (SCN) is the interaction between its two components (see Fig. 1). The first component takes the image as input to generate locally accurate but potentially ambiguous local appearance heatmaps $h_{i}^{LA}(\pmb{x})$. Motivated by handcrafted graphical models for eliminating these potential ambiguities, the second component takes the predicted candidate heatmaps $h_{i}^{LA}(\pmb{x})$ as input to generate inaccurate
but unambiguous spatial configuration heatmaps $h_{i}^{SC}(\pmb{x})$.

For $N$ landmarks, the set of predicted heatmaps $\mathbb{H} = \{h_{i}(\pmb{x}) | i = 1 \cdots N\}$ is obtained by element-wise multiplication $\odot$ of the corresponding heatmap outputs $h_{i}^{LA}(\pmb{x})$ and  $h_{i}^{SC}(\pmb{x})$ of the two components:
$$
h_{i}(\pmb{x}) = h_{i}^{LA}(\pmb{x}) \odot h_{i}^{SC}(\pmb{x}) \tag{1}
$$



This multiplication is crucial for the SCN, as it forces both of its components to generate a response on the location of the target landmark $\pmb{x}^*_{i}$, i.e., both  $h_{i}^{LA}(\pmb{x})$ and  $h_{i}^{SC}(\pmb{x})$deliver responses for x close to $\pmb{x}^*_{i}$ , while on all other locations one component may have a response as long as the other one does not have one.  

通过这篇论文 《Integrating Spatial Configuration into Heatmap Regression Based CNNs for Landmark Localization》只可以知道输出结果由2部分组成，分别是 local appearance heatmaps 和spatial configuration heatmaps。输出结果为二者的逐点乘积。

```python
heatmap = HLA * HSC
```

关于HLA和HSC的计算没有明确说明。其中HLA和HSC共用一个基础的Unet，不同之处在于HLA生成

参考 [MedicalDataAugmentationTool-VerSe/network.py at master · christianpayer/MedicalDataAugmentationTool-VerSe (github.com)](https://github.com/christianpayer/MedicalDataAugmentationTool-VerSe/blob/master/verse2019/network.py)

《Coarse to Fine Vertebrae Localization and Segmentation with SpatialConfiguration-Net and U-Net》

参考实现（论文作者的tensorflow实现）：

```python
    scnet_local = actual_network(num_filters_base=num_filters_base,
                                 num_levels=4,
                                 double_filters_per_level=False,
                                 normalization=None,
                                 activation=activation,
                                 data_format=data_format,
                                 padding=padding)
    unet_out = scnet_local(node, is_training)
    local_heatmaps = conv3d(unet_out,
                            filters=num_labels,
                            kernel_size=[3, 3, 3],
                            name='local_heatmaps',
                            kernel_initializer=heatmap_layer_kernel_initializer,
                            activation=None,
                            data_format=data_format,
                            is_training=is_training)
    downsampled = avg_pool3d(local_heatmaps, [downsampling_factor] * 3, name='local_downsampled', data_format=data_format)
    conv = conv3d(downsampled, filters=num_filters_base, kernel_size=[7, 7, 7], name='sconv0', activation=activation, data_format=data_format, is_training=is_training, padding=padding)
    conv = conv3d(conv, filters=num_filters_base, kernel_size=[7, 7, 7], name='sconv1', activation=activation, data_format=data_format, is_training=is_training, padding=padding)
    conv = conv3d(conv, filters=num_filters_base, kernel_size=[7, 7, 7], name='sconv2', activation=activation, data_format=data_format, is_training=is_training, padding=padding)
    conv = conv3d(conv, filters=num_labels, kernel_size=[7, 7, 7], name='spatial_downsampled', kernel_initializer=heatmap_layer_kernel_initializer, activation=tf.nn.tanh, data_format=data_format, is_training=is_training, padding=padding)
    if data_format == 'channels_last':
        # suppose that 'channels_last' means CPU
        # resize_trilinear is much faster on CPU
        spatial_heatmaps = resize_tricubic(conv, factors=[downsampling_factor] * 3, name='spatial_heatmaps', data_format=data_format)
    else:
        # suppose that 'channels_first' means GPU
        # upsample3d_linear is much faster on GPU
        spatial_heatmaps = upsample3d_cubic(conv, factors=[downsampling_factor] * 3, name='spatial_heatmaps', data_format=data_format, padding='valid_cropped')

    heatmaps = local_heatmaps * spatial_heatmaps

    return heatmaps, local_heatmaps, spatial_heatmaps
```



[MedicalDataAugmentationTool-HeatmapRegression/network.py at master · christianpayer/MedicalDataAugmentationTool-HeatmapRegression (github.com)](https://github.com/christianpayer/MedicalDataAugmentationTool-HeatmapRegression/blob/master/spine/network.py)

该项目中也有关于SCN的实现

```python


import tensorflow as tf
from tensorflow_train.layers.layers import conv3d, avg_pool3d, dropout, add
from tensorflow_train.layers.interpolation import upsample3d_linear, upsample3d_cubic
from tensorflow_train.networks.unet_base import UnetBase
from tensorflow_train.networks.unet import UnetClassic3D


class SCNetLocal(UnetBase):
    def downsample(self, node, current_level, is_training):
        return avg_pool3d(node, [2, 2, 2], name='downsample' + str(current_level), data_format=self.data_format)

    def upsample(self, node, current_level, is_training):
        return upsample3d_linear(node, [2, 2, 2], name='upsample' + str(current_level), data_format=self.data_format)

    def conv(self, node, current_level, postfix, is_training):
        return conv3d(node,
                      self.num_filters(current_level),
                      [3, 3, 3],
                      name='conv' + postfix,
                      activation=self.activation,
                      normalization=self.normalization,
                      is_training=is_training,
                      data_format=self.data_format,
                      padding=self.padding)

    def combine(self, parallel_node, upsample_node, current_level, is_training):
        return add([parallel_node, upsample_node], name='add' + str(current_level))

    def contracting_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '_0', is_training)
        node = dropout(node, 0.5, 'drop' + str(current_level), is_training)
        node = self.conv(node, current_level, '_1', is_training)
        return node

    def parallel_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '', is_training)
        return node

    def expanding_block(self, node, current_level, is_training):
        return node


def network_scn(input, num_heatmaps, is_training, data_format='channels_first'):
    num_filters_base = 64
    activation = lambda x, name: tf.nn.leaky_relu(x, name=name, alpha=0.1)
    padding = 'reflect'
    heatmap_layer_kernel_initializer = tf.truncated_normal_initializer(stddev=0.001)
    downsampling_factor = 8
    node = conv3d(input,
                  filters=num_filters_base,
                  kernel_size=[3, 3, 3],
                  name='conv0',
                  activation=activation,
                  data_format=data_format,
                  is_training=is_training)
    scnet_local = SCNetLocal(num_filters_base=num_filters_base,
                             num_levels=4,
                             double_filters_per_level=False,
                             normalization=None,
                             activation=activation,
                             data_format=data_format,
                                      padding=padding)
    unet_out = scnet_local(node, is_training)
    local_heatmaps = conv3d(unet_out,
                            filters=num_heatmaps,
                            kernel_size=[3, 3, 3],
                            name='local_heatmaps',
                            kernel_initializer=heatmap_layer_kernel_initializer,
                            activation=None,
                            data_format=data_format,
                            is_training=is_training)
    downsampled = avg_pool3d(local_heatmaps, [downsampling_factor] * 3, name='local_downsampled', data_format=data_format)
    conv = conv3d(downsampled, filters=num_filters_base, kernel_size=[7, 7, 7], name='sconv0', activation=activation, data_format=data_format, is_training=is_training, padding=padding)
    conv = conv3d(conv, filters=num_filters_base, kernel_size=[7, 7, 7], name='sconv1', activation=activation, data_format=data_format, is_training=is_training, padding=padding)
    conv = conv3d(conv, filters=num_filters_base, kernel_size=[7, 7, 7], name='sconv2', activation=activation, data_format=data_format, is_training=is_training, padding=padding)
    conv = conv3d(conv, filters=num_heatmaps, kernel_size=[7, 7, 7], name='spatial_downsampled', kernel_initializer=heatmap_layer_kernel_initializer, activation=tf.nn.tanh, data_format=data_format, is_training=is_training, padding=padding)
    spatial_heatmaps = upsample3d_cubic(conv, [downsampling_factor] * 3, name='spatial_heatmaps', data_format=data_format, padding='valid_cropped')

    heatmaps = local_heatmaps * spatial_heatmaps

    return heatmaps, local_heatmaps, spatial_heatmaps


def network_unet(input, num_heatmaps, is_training, data_format='channels_first'):
    num_filters_base = 64
    activation = tf.nn.relu
    node = conv3d(input,
                  filters=num_filters_base,
                  kernel_size=[3, 3, 3],
                  name='conv0',
                  activation=activation,
                  data_format=data_format,
                  is_training=is_training)
    scnet_local = UnetClassic3D(num_filters_base=num_filters_base,
                             num_levels=5,
                             double_filters_per_level=False,
                             normalization=None,
                             activation=activation,
                             data_format=data_format)
    unet_out = scnet_local(node, is_training)
    heatmaps = conv3d(unet_out,
                            filters=num_heatmaps,
                            kernel_size=[3, 3, 3],
                            name='heatmaps',
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.0001),
                            activation=None,
                            data_format=data_format,
                            is_training=is_training)

    return heatmaps, heatmaps, heatmaps
```

### SCN其他实现1



```python
import torch
import torch.nn as nn
from typing import Sequence


ACT = {'relu': nn.ReLU, 'leaky': nn.LeakyReLU, 'prelu': nn.PReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}


class SCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        filters: int = 64,
        factor: int = 8,
        dropout: float = 0.,
        mode: str = 'add',
        local_act: str = None,
        spatial_act: str = 'tanh',
    ):
        super().__init__()
        self.HLA = LocalAppearance(in_channels, num_classes, filters, dropout, mode)
        self.down = nn.AvgPool3d(factor, factor, ceil_mode=True)
        self.up = nn.Upsample(scale_factor=factor, mode='trilinear', align_corners=True)
        self.local_act = ACT[local_act]() if local_act else None
        self.HSC = nn.Sequential(
            nn.Conv3d(filters, filters, 7, 1, 3, bias=False),
            nn.Conv3d(filters, filters, 7, 1, 3, bias=False),
            nn.Conv3d(filters, filters, 7, 1, 3, bias=False),
            nn.Conv3d(filters, num_classes, 7, 1, 3, bias=False),
        )
        self.spatial_act = ACT[spatial_act]()
        nn.init.trunc_normal_(self.HSC[-1].weight, 0, 1e-4)

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        d1, HLA = self.HLA(x)
        if self.local_act:
            HLA = self.local_act(HLA)
        HSC = self.up(self.spatial_act(self.HSC(self.down(d1))))
        heatmap = HLA * HSC
        return heatmap, HLA, HSC


class LocalAppearance(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        filters: int = 64,
        dropout: float = 0.,
        mode: str = 'add',
    ):
        super().__init__()
        self.mode = mode
        self.pool = nn.AvgPool3d(2, 2, ceil_mode=True)
        self.up = nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=True)
        self.in_conv = self.Block(in_channels, filters)
        self.enc1 = self.Block(filters, filters, dropout)
        self.enc2 = self.Block(filters, filters, dropout)
        self.enc3 = self.Block(filters, filters, dropout)
        self.enc4 = self.Block(filters, filters, dropout)
        if mode == 'add':
            self.dec3 = self.Block(filters, filters, dropout)
            self.dec2 = self.Block(filters, filters, dropout)
            self.dec1 = self.Block(filters, filters, dropout)
        else:
            self.dec3 = self.Block(2*filters, filters, dropout)
            self.dec2 = self.Block(2*filters, filters, dropout)
            self.dec1 = self.Block(2*filters, filters, dropout)
        self.out_conv = nn.Conv3d(filters, num_classes, 1, bias=False)
        nn.init.trunc_normal_(self.out_conv.weight, 0, 1e-4)

    def Block(self, in_channels, out_channels, dropout=0):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.Dropout3d(dropout, True),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.Dropout3d(dropout, True),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.in_conv(x)
        e1 = self.enc1(x0)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        if self.mode == 'add':
            d3 = self.dec3(self.up(e4)+e3)
            d2 = self.dec2(self.up(d3)+e2)
            d1 = self.dec1(self.up(d2)+e1)
        else:
            d3 = self.dec3(torch.cat([self.up(e4), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))

        out = self.out_conv(d1)
        return d1, out


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        filters: int = 64,
    ) -> None:
        super().__init__()
        self.in_conv = self.Block(in_channels, filters)
        self.enc1 = self.Block(filters, filters)
        self.enc2 = self.Block(filters, filters)
        self.enc3 = self.Block(filters, filters)
        self.enc4 = self.Block(filters, filters)
        self.enc5 = self.Block(filters, filters)
        self.dec4 = self.Block(2*filters, filters)
        self.dec3 = self.Block(2*filters, filters)
        self.dec2 = self.Block(2*filters, filters)
        self.dec1 = self.Block(2*filters, filters)
        self.out_conv = nn.Conv3d(filters, num_classes, 1, bias=False)
        self.pool = nn.AvgPool3d(2, 2, ceil_mode=True)
        self.up = nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.in_conv(x)
        e1 = self.enc1(x0)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up(e5), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))

        out = self.out_conv(d1)
        return torch.sigmoid(out)

    def Block(self, in_channels, out_channels, dropout=0):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.Dropout3d(dropout, True),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.Dropout3d(dropout, True),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
```



### SCN其他实现2



```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class UNet(nn.Module):
    def __init__(self, channels, classes):
        super().__init__()
        self.conv = DoubleConv(channels, 16, 32)
        self.down1 = Down(32, 32, 64)
        self.down2 = Down(64, 64, 128)
        self.down3 = Down(128, 128, 256)

        self.up1 = Up(384, 128, 128)
        self.up2 = Up(192, 64, 64)
        self.up3 = Up(96, 32, 32)
        self.out = OutConv(32, classes)
        self.spacial = Spacial_Info_Tanh(32, classes)
        #self.spacial = Spacial_Info(32, classes).cuda(0)
        #self.norm = nn.ReLU(inplace=True)

        #self.edge = EdgeOut(32, 26).cuda(1)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        x = self.up1(x3,x2)
        x = self.up2(x,x1)
        x = self.up3(x,x0)
        #edge = self.edge(x)

        local = self.out(x)

        spacial = self.spacial(x)
        x = local * spacial

        # for check
        # self.localt = local
        # self.spacialt = spacial
        #x = self.norm(x)
        return x
        # return [x, edge]

class SCN_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet(channels=1,classes=15)
        self.sccov0 = nn.Conv3d(in_channels=15,out_channels=64,kernel_size=[7,7,7],padding=3)
        self.sccov1 = nn.Conv3d(in_channels=64,out_channels=64,kernel_size=[7,7,7],padding=3)
        self.sccov2 = nn.Conv3d(in_channels=64,out_channels=64,kernel_size=[7,7,7],padding=3)
        self.sccov3 = nn.Conv3d(in_channels=64,out_channels=15,kernel_size=[7,7,7],padding=3)
        self.downsample = nn.MaxPool3d(kernel_size=4)
        self.upsample = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)

    def forward(self,x):
        local_pred = self.unet(x).cuda(1)
        spatial_pred = self.downsample(local_pred)
        spatial_pred = self.sccov0(spatial_pred)
        spatial_pred = self.sccov1(spatial_pred)
        spatial_pred = self.sccov2(spatial_pred)
        spatial_pred = self.sccov3(spatial_pred)
        spatial_pred = self.upsample(spatial_pred)
        prediction = spatial_pred*local_pred
        return prediction
    
  class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channel1, out_channel2):
        super().__init__()
        self.doubleConv = nn.Sequential(
            nn.Conv3d(in_channels, out_channel1, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channel1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channel1, out_channel2, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channel2),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.doubleConv(x)

class EdgeOut(nn.Module):
    def __init__(self, in_channels, edge_num):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, padding=3)
        self.avgpool = nn.AvgPool3d(3)
        self.linear = nn.Linear(42 * 42 * 66, edge_num * 3)
        self.edge_num = edge_num

    def forward(self, x):
        x = self.conv1(x)
        x = self.avgpool(x)
        x = torch.max(x, dim=1)[0]
        x = torch.reshape(x, (1, 42 * 42 * 66))
        x = self.linear(x)
        x = torch.reshape(x, (1, self.edge_num, 3))
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channel1, out_channel2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, padding=0),
            DoubleConv(in_channels, out_channel1, out_channel2)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channel1, out_channel2):
        super().__init__()
        self.up =nn.Upsample(scale_factor=2.0, mode="trilinear")
        self.conv = DoubleConv(in_channels, out_channel2, out_channel2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        nn.init.normal_(self.conv.weight, 0, 0.0001)
        nn.init.constant_(self.conv.bias, 0)
        # self.conv = nn.Conv3d(in_channels, out_channels,kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class Spacial_Info(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Upsample(scale_factor=0.25, mode="trilinear")
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=7, padding=3)
        self.conv3 = nn.Conv3d(64, 64, kernel_size=7, padding=3)

        self.conv4 = nn.Conv3d(64, out_channels, kernel_size=7, padding=3)
        self.conv5 = nn.Conv3d(out_channels, out_channels, kernel_size=7, padding=3)
        self.conv6 = nn.Conv3d(out_channels, out_channels, kernel_size=7, padding=3)
        nn.init.normal_(self.conv6.weight, 0, 0.0001)
        nn.init.constant_(self.conv6.bias, 0)
        self.softmax = nn.Softmax()
        self.relu = nn.LeakyReLU(inplace=True)


    def forward(self, x):
        a, b, c = x.shape[2:]
        x = self.down(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.softmax(x)
        self.up = nn.Upsample((a, b, c), mode="trilinear")
        x = self.up(x)
        return x

class Spacial_Info_Tanh(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Upsample(scale_factor=0.25, mode="trilinear")
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=7, padding=3)
        self.conv3 = nn.Conv3d(64, 64, kernel_size=7, padding=3)

        self.conv4 = nn.Conv3d(64, out_channels, kernel_size=7, padding=3)
        self.conv5 = nn.Conv3d(out_channels, out_channels, kernel_size=7, padding=3)
        self.conv6 = nn.Conv3d(out_channels, out_channels, kernel_size=7, padding=3)
        nn.init.normal_(self.conv6.weight, 0, 0.0001)
        nn.init.constant_(self.conv6.bias, 0)
        self.tanh = nn.Tanh()

    def forward(self, x):
        a, b, c = x.shape[2:]
        x = self.down(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.tanh(x)
        self.up = nn.Upsample((a, b, c), mode="trilinear")
        x = self.up(x)
        return x



```







## pytorch中图像的分块（patch）操作



[(1条消息) pytorch中图像的分块（patch）操作（使用了.permute()维度转换）_cdy艳0917的博客-CSDN博客_patch pytorch](https://blog.csdn.net/sinat_42239797/article/details/104666149)





























# 深度学习训练网络通用架构

## pytorch



```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# 定义网络结构
class Model(nn.Module):
	def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))

# 网络实例化
model = Model()
# loss
loss_func = F.cross_entropy
# 优化器设置
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False)

for epoch in range(epochs):
	# 设置为训练模式
    model.train()
    # iterate: 每次一个batch
    for xb, yb in train_dl:
    	# 前向传播
        pred = model(xb)
        # 计算损失
        loss = loss_func(pred, yb)
		# 反向传播，计算loss关于各权重参数的偏导，更新grad
        loss.backward()
        # 优化器基于梯度下降原则，更新（学习）权重参数parameters
        opt.step()
        # 各权重参数的偏导清零 grad=>0
        opt.zero_grad()
	# 设置为评估（推理）模式，设置BN、dropout等模块
    model.eval()
    # 不更新梯度
    with torch.no_grad():
        valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

    print(epoch, valid_loss / len(valid_dl))

```



### 数据集加载

#### DataLoader

At the heart of PyTorch data loading utility is the [`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader) class. It represents a Python iterable over a dataset, with support for

- [map-style and iterable-style datasets](https://pytorch.org/docs/stable/data.html?highlight=dataloader#dataset-types),
- [customizing data loading order](https://pytorch.org/docs/stable/data.html?highlight=dataloader#data-loading-order-and-sampler),
- [automatic batching](https://pytorch.org/docs/stable/data.html?highlight=dataloader#loading-batched-and-non-batched-data),
- [single- and multi-process data loading](https://pytorch.org/docs/stable/data.html?highlight=dataloader#single-and-multi-process-data-loading),
- [automatic memory pinning](https://pytorch.org/docs/stable/data.html?highlight=dataloader#memory-pinning).

These options are configured by the constructor arguments of a [`DataLoader`](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.DataLoader), which has signature:

```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
```



#### Dataset

*CLASS*`torch.utils.data.``Dataset`(**args*, ***kwds*) [[SOURCE](https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Dataset)]

An abstract class representing a [`Dataset`](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset).

All datasets that represent a map from keys to data samples should subclass it. All subclasses should overwrite `__getitem__()`, supporting fetching a data sample for a given key. Subclasses could also optionally overwrite `__len__()`, which is expected to return the size of the dataset by many [`Sampler`](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Sampler) implementations and the default options of [`DataLoader`](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.DataLoader).

##### 代码中的例子

```python
# -*- coding: utf-8 -*-
import torch
import torch.utils.data as Data
torch.manual_seed(1)    # reproducible
 
BATCH_SIZE = 5
 
x = torch.linspace(1, 10, 10)       # this is x data (torch tensor)
y = torch.linspace(10, 1, 10)       # this is y data (torch tensor)
 
'''先转换成 torch 能识别的 Dataset'''
torch_dataset = Data.TensorDataset(x, y)
print(torch_dataset[0])     #输出(tensor(1.), tensor(10.))
print(torch_dataset[1])     #输出(tensor(2.), tensor(9.))
 
''' 把 dataset 放入 DataLoader'''
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    #num_workers=2,              # subprocesses for loading data
)
 
for epoch in range(3):   # train entire dataset 3 times
    for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
        # train your data...
        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())
```

运行输出：

```bash
(tensor(1.), tensor(10.))
(tensor(2.), tensor(9.))
Epoch:  0 | Step:  0 | batch x:  [5. 3. 1. 7. 9.] | batch y:  [ 6.  8. 10.  4.  2.]
Epoch:  0 | Step:  1 | batch x:  [ 8. 10.  2.  6.  4.] | batch y:  [3. 1. 9. 5. 7.]
Epoch:  1 | Step:  0 | batch x:  [5. 9. 2. 6. 1.] | batch y:  [ 6.  2.  9.  5. 10.]
Epoch:  1 | Step:  1 | batch x:  [ 3.  4.  7. 10.  8.] | batch y:  [8. 7. 4. 1. 3.]
Epoch:  2 | Step:  0 | batch x:  [6. 9. 4. 8. 7.] | batch y:  [5. 2. 7. 3. 4.]
Epoch:  2 | Step:  1 | batch x:  [10.  3.  2.  1.  5.] | batch y:  [ 1.  8.  9. 10.  6.]
```



##### 参考例子1



```python
class SpineCentroidSet(Dataset):
    def __init__(self, root: str, files: List[str], mode: str = 'train', norm: bool = True, num_classes: int = 25) -> None:
        super().__init__()
        self.files = [os.path.join(root, os.path.basename(p)) for p in files]
        self.mode = mode
        self.num_classes = num_classes
        self.norm = norm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        label_path = self.files[index]
        image_path = label_path.replace("_seg", "")
        image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        if self.mode == 'train':
            image, label = random_crop(image, label)
        image = normalize(image.astype(np.float32)) if self.norm else image
        if self.num_classes > 1:
            landmark = generate_landmark(label, self.num_classes)
        else:
            landmark = generate_one_channel_landmark(label)
        image = image[np.newaxis, ...]
        label = label[np.newaxis, ...]
        return image, label, landmark
    
def normalize(img: np.ndarray):
    '''Intensity value of the CT volumes is divided by 2048 and clamped between -1 and 1'''
    return np.clip(img / 2048, -1, 1)


def random_crop(image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''在2mm脊柱图像(尺寸为[Z, 96, 96])中随机裁剪[128, 96, 96]大小'''
    if label.shape == (128, 96, 96):
        return image, label

    # np.random.randint是[low, high)区间
    z_range = (64, label.shape[0] - 64 + 1)
    label_cropped = np.zeros((128, 96, 96), dtype=np.uint8)
    classes = np.unique(label_cropped)
    while(len(classes) == 1):
        z = np.random.randint(*z_range)
        label_cropped = label[z-64:z+64]
        classes = np.unique(label_cropped)
        # 在VerSe数据集中, 如果椎骨不完整是没有标注的
        if len(classes) > 1:
            # TODO 可以修改成按百分比来计算
            if(label_cropped == classes[1]).sum() != (label == classes[1]).sum():
                # 删除顶端不完整的标注
                label_cropped[label_cropped == classes[1]] = 0

            if(label_cropped == classes[-1]).sum() != (label == classes[-1]).sum():
                # 删除底端不完整的标注
                label_cropped[label_cropped == classes[-1]] = 0
        classes = np.unique(label_cropped)
    image_cropped = image[z-64:z+64]
    return image_cropped, label_cropped


def generate_landmark(label: np.ndarray, num: int) -> np.ndarray:
    '''Generate one-hot landmark volume'''
    landmark = np.zeros((num, *label.shape), np.float32)
    classes = np.unique(label)
    for c in range(1, num+1):
        if c in classes:
            Z, Y, X = np.where(label == c)
            Z = np.round(Z.mean()).astype(int)
            Y = np.round(Y.mean()).astype(int)
            X = np.round(X.mean()).astype(int)
            landmark[c-1, Z, Y, X] = 1
    return landmark


def generate_one_channel_landmark(label: np.ndarray) -> np.ndarray:
    '''Generate a single channel landmark volume'''
    landmark = np.zeros(label.shape, np.float32)
    classes = np.unique(label)
    for c in classes[1:]:
        Z, Y, X = np.where(label == c)
        Z = np.round(Z.mean()).astype(int)
        Y = np.round(Y.mean()).astype(int)
        X = np.round(X.mean()).astype(int)
        landmark[Z, Y, X] = 1
    return landmark[np.newaxis, ...]

```



##### 参考例子2



```python
class SegDataset(Dataset):
    def __init__(
        self,
        root,
        file_list: List[str],
        mode: str = 'train',
        patch_size: Tuple[int] = (128, 128, 96),
        augment: bool = False,
        weight: bool = False
    ):
        super().__init__()
        self.root = root
        self.file_list = file_list
        self.mode = mode
        self.patch_size = patch_size
        self.augment = augment
        self.weight = weight

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if self.mode == 'train':
            return self._get_train_data(index)
        return self._get_inf_data(index)

    def _get_train_data(self, index):
        path = self.file_list[index]
        basename = os.path.basename(path)
        ID = basename[:basename.find("_seg.nii.gz")]
        mask_path = os.path.join(self.root, basename)
        img_path = mask_path.replace("_seg", "")
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        image = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        image, mask, landmark, category = self.generate_random_patch(image, mask)
        image = normalize(image.astype(np.float32))
        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        landmark = np.expand_dims(landmark, 0)
        return ID, image, mask, landmark, category

    def _get_inf_data(self, index):
        path = self.file_list[index]
        basename = os.path.basename(path)
        ID = basename[:basename.find("_seg.nii.gz")]
        mask_path = os.path.join(self.root, basename)
        img_path = mask_path.replace("_seg", "")
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        image = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        classes = np.unique(mask)
        return ID, image, mask, classes[1:]

    def get_bbox(self, shape: np.ndarray, center: tuple):
        patch_size = list(self.patch_size)[::-1]
        # 原始图像的bbox | original bbox
        z_min = max(0, center[0] - patch_size[0]//2)
        z_max = min(shape[0], center[0] + patch_size[0]//2)
        y_min = max(0, center[1] - patch_size[1]//2)
        y_max = min(shape[1], center[1] + patch_size[1]//2)
        x_min = max(0, center[2] - patch_size[2]//2)
        x_max = min(shape[2], center[2] + patch_size[2]//2)

        # 新图像的bbox | new bbox
        Z_MIN = patch_size[0]//2 - (center[0] - z_min)
        Z_MAX = patch_size[0]//2 + (z_max - center[0])
        Y_MIN = patch_size[1]//2 - (center[1] - y_min)
        Y_MAX = patch_size[1]//2 + (y_max - center[1])
        X_MIN = patch_size[2]//2 - (center[2] - x_min)
        X_MAX = patch_size[2]//2 + (x_max - center[2])
        return (z_min, z_max, y_min, y_max, x_min, x_max), (Z_MIN, Z_MAX, Y_MIN, Y_MAX, X_MIN, X_MAX)

    def generate_random_patch(self, image, mask):
        classes = np.unique(mask)[1:]
        idx = random.randint(0, len(classes)-1)
        Z, Y, X = np.where(mask == classes[idx])
        Z = Z.mean().round().astype(int)
        Y = Y.mean().round().astype(int)
        X = X.mean().round().astype(int)
        patch_size = list(self.patch_size)[::-1]
        bbox, BBOX = self.get_bbox(mask.shape, (Z, Y, X))

        new_mask = np.zeros(patch_size, dtype=np.uint8)
        new_img = -1023*np.ones(patch_size, dtype=np.int16)
        landmark = np.zeros(patch_size, dtype=np.float32)
        landmark[tuple([shape//2 for shape in patch_size])] = 1
        new_mask[
            BBOX[0]:BBOX[1],
            BBOX[2]:BBOX[3],
            BBOX[4]:BBOX[5]
        ] = mask[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
        new_img[
            BBOX[0]:BBOX[1],
            BBOX[2]:BBOX[3],
            BBOX[4]:BBOX[5]
        ] = image[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
        new_mask[new_mask != classes[idx]] = 0
        new_mask[new_mask > 0] = 1
        if self.augment:
            new_img,new_mask = elastic_transformation(new_img,new_mask)
        return new_img, new_mask, landmark, classes[idx]

    def generate_inf_patch(self, image, mask):
        img_list = []
        mask_list = []
        patch_size = list(self.patch_size)[::-1]
        classes = np.unique(mask)[1:]
        for c in classes:
            new_mask = np.zeros(patch_size, dtype=np.uint8)
            new_img = -1023*np.ones(patch_size, dtype=np.int16)
            Z, Y, X = np.where(mask == c)
            Z = Z.mean().round().astype(int)
            Y = Y.mean().round().astype(int)
            X = X.mean().round().astype(int)
            bbox, BBOX = self.get_bbox(mask.shape, (Z, Y, X))
            new_mask[
                BBOX[0]:BBOX[1],
                BBOX[2]:BBOX[3],
                BBOX[4]:BBOX[5]
            ] = mask[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
            new_img[
                BBOX[0]:BBOX[1],
                BBOX[2]:BBOX[3],
                BBOX[4]:BBOX[5]
            ] = image[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
            new_mask[new_mask != c] = 0
            new_mask[new_mask > 0] = 1

            img_list.append(np.expand_dims(new_img, 0))
            mask_list.append(np.expand_dims(new_mask, 0))

        new_img = torch.Tensor(np.vstack(img_list)).unsqueeze(
            1)  # [N,1,96,128,128]
        new_img = torch.clip(new_img/2048, -1, 1)
        new_mask = torch.Tensor(np.vstack(mask_list)).unsqueeze(1)
        landmark = torch.zeros_like(new_mask, dtype=torch.float32)
        landmark[:, 0, patch_size[0]//2,
                 patch_size[1]//2, patch_size[2]//2] = 1
        return TensorDataset(new_img, new_mask, landmark)

def elastic_transformation(img: np.ndarray, mask: np.ndarray, alpha: float = 20, sigma: float = 5):
    '''Elastic deformation of images as described in [Simard2003]_.'''
    shape = img.shape
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    z, y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    coordinates = np.reshape(z + dz, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    distorted_img = map_coordinates(img, coordinates, None, 3, 'reflect').reshape(shape)
    distorted_mask = map_coordinates(mask, coordinates, None, 1, 'reflect').reshape(shape)
    return distorted_img, distorted_mask

```



##### 参考例子3



```python
import os
import torch
import numba as nb
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
from typing import List


cls_map = {
    11: 0, 12: 1, 13: 2, 14: 3, 15: 4, 16: 5, 17: 6, 18: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15,
    31: 16, 32: 17, 33: 18, 34: 19, 35: 20, 36: 21, 37: 22, 38: 23,
    41: 24, 42: 25, 43: 26, 44: 27, 45: 28, 46: 29, 47: 30, 48: 31
}


class ToothSet(Dataset):
    def __init__(self, file_list: List[str], num_class: int, augment_prob: float, detection: bool,  offset: bool):
        super().__init__()
        self.file_list = file_list
        self.augment_prob = augment_prob
        self.num_class = num_class
        if detection and offset:
            raise ValueError("You can only select one type!")
        self.detection = detection
        self.offset = offset
        self.classmap = {v: k for k, v in cls_map.items()}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        image_path = self.file_list[index]
        label_path = image_path.replace("image", "label")
        basename = os.path.basename(image_path)
        image = sitk.ReadImage(image_path)
        label = sitk.ReadImage(label_path)
        origin = self.get_origin(label)
        if np.random.uniform() > self.augment_prob:
            angle = np.random.randint(-10, 0)
            radian = angle/180 * np.pi
            offset = [
                np.random.randint(-5, 5)*0.5,
                np.random.randint(-5, 5)*0.5,
                np.random.randint(-5, 5)*0.5
            ]
            center_ijk = np.array(image.GetSize())//2
            center_ras = image.TransformContinuousIndexToPhysicalPoint(
                center_ijk.tolist())
            t = sitk.AffineTransform(3)
            t.SetCenter(center_ras)
            t.Rotate(1, 2, radian)
            t.Translate(offset)
        else:
            t = sitk.Transform(3, sitk.sitkIdentity)
        image = self.transform(image, origin, t, sitk.sitkLinear, -1024)
        label = self.transform(label, origin, t, sitk.sitkNearestNeighbor, 0)
        # sitk.WriteImage(image, f"test/image/{basename}")
        # sitk.WriteImage(label, f"test/label/{basename}")
        image = sitk.GetArrayFromImage(image)
        label = sitk.GetArrayFromImage(label)
        if self.detection:
            label = self.generate_landmark(label)
        image = image[np.newaxis, :, :, :]
        if self.offset:
            label = label[np.newaxis, :, :, :]
            seg = (label > 0).astype(np.uint8)
            offset = generate_offset(label)
            return image, (seg, offset)
        return image, label

    def get_origin(self, sitk_label: sitk.Image) -> np.ndarray:
        '''计算裁剪图像的origin'''
        size = sitk_label.GetSize()
        index = np.array(size)//2
        center = np.array(
            sitk_label.TransformContinuousIndexToPhysicalPoint(index.tolist()))

        '''训练集中有两类数据
        ①一类的数据尺寸为80mm*80mm*80mm,这类数据FOV很小,可以直接将裁剪中心置为CT中心
        ②另一类数据尺寸通常为160mm*160mm*87mm,这类数据FOV较大,需要对中心点进行偏移
        '''
        if(size != (160, 160, 160)):
            label = sitk.GetArrayFromImage(sitk_label)
            Z, _, _ = np.where(label > 0)
            index = np.array([
                size[0]//2,
                (size[1]//2-144 + size[1]//2 + 48)//2,
                (Z.min() + Z.max())//2
            ])

            center = np.array(
                sitk_label.TransformContinuousIndexToPhysicalPoint(index.tolist()))
        origin = center - 96*0.5
        return origin

    def transform(self, image: sitk.Image, origin: tuple, t: sitk.Transform, method: int, fillValue: int = -1024) -> sitk.Image:
        filter = sitk.ResampleImageFilter()
        filter.SetInterpolator(method)
        filter.SetTransform(t)
        filter.SetSize((192, 192, 192))
        filter.SetOutputOrigin(origin)
        filter.SetOutputDirection(image.GetDirection())
        filter.SetOutputSpacing(image.GetSpacing())
        filter.SetDefaultPixelValue(fillValue)
        return filter.Execute(image)

    def generate_landmark(self, label: np.ndarray) -> np.ndarray:
        '''生成牙齿质心点'''
        landmark = np.zeros(label.shape, np.float32)
        classes = np.unique(label)
        for c in classes[1:]:
            Z, Y, X = np.where(label == c)
            Z = np.round(Z.mean()).astype(int)
            Y = np.round(Y.mean()).astype(int)
            X = np.round(X.mean()).astype(int)
            landmark[Z, Y, X] = 1
        return landmark[np.newaxis, :, :, :]


@nb.jit(nopython=True, cache=True)
def generate_offset(label: np.ndarray):
    '''使用numba加速偏移值的计算'''
    classes = np.unique(label)
    offset = np.zeros((3, 192, 192, 192), nb.int16)
    for c in classes[1:]:
        _, Z, Y, X = np.where(label == c)
        centroid_z = np.round(Z.mean())
        centroid_y = np.round(Y.mean())
        centroid_x = np.round(X.mean())
        for j in range(len(Z)):
            offset[0, Z[j], Y[j], X[j]] = centroid_z - Z[j]
            offset[1, Z[j], Y[j], X[j]] = centroid_y - Y[j]
            offset[2, Z[j], Y[j], X[j]] = centroid_x - X[j]
    return offset
```







### 损失函数



### 训练





## tensorflow





## paddle







# pytorch基础知识



## torchvision



torchvision是pytorch的一个图形库，它服务于PyTorch深度学习框架的。其构成如下：

torchvision.datasets: 一些加载数据的函数及常用的数据集接口；
torchvision.models: 包含常用的模型结构（含预训练模型），例如AlexNet、VGG、ResNet等；
torchvision.transforms: 常用的图片变换，例如裁剪、旋转等；
torchvision.utils: 其他的一些有用的方法。




## 训练技巧

[PyTorch学习记录——PyTorch进阶训练技巧_maximejia的博客-CSDN博客_pytorch 训练](https://blog.csdn.net/maximejia/article/details/125902794?spm=1001.2101.3001.6650.17&utm_medium=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-17-125902794-blog-123258204.pc_relevant_aa&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-17-125902794-blog-123258204.pc_relevant_aa&utm_relevant_index=17)



### 自定义损失函数



自定义损失函数的方法主要包括两种，即以函数的方式定义和以类的方式定义。

#### 以函数的方式定义损失函数

以函数的方式定义与定义python函数没有什么区别，通过将参与损失计算的张量（即Tensor）作为函数的形参进行定义，例如

```python
def my_loss(output: torch.Tensor, target: torch.Tensor):
    loss = torch.mean((output - target) ** 2)
    return loss
```

在上述定义中，我们使用了MSELoss损失函数。同时可以看到，在损失函数编写过程中，可以直接使用我们熟悉的Python中的运算符，包括加减乘除等等，但牵涉到矩阵运算，如矩阵乘法则需要使用Pytorch提供的张量计算接口`torch.matmul`。采用这样的方式定义损失函数实际上就仅需要把计算过程定义清楚即可，或者说是把计算图或数据流定义清楚。

#### 以类的方式定义损失函数

以类的方式定义损失函数需要让我们定义的类继承`nn.Module`类。采用这样的方式定义损失函数类，可以让我们把定义的损失函数作为一个神经网络层来对待。Pytorch现有的损失函数也大都采用这种类的方式进行定义的。事实上，在Pytorch中，`Loss`函数部分继承自`_loss`, 部分继承自`_WeightedLoss`, 而`_WeightedLoss`继承自`_loss`，` _loss`继承自 `nn.Module`。例如，通过查看Pytorch中`CrossEntropyLoss`的代码，我们可以看到上述关系，如下。


```python
class CrossEntropyLoss(_WeightedLoss):
        ...

class _WeightedLoss(_Loss):
    ...

class _Loss(Module):
    ...

```

#### 比较与思考



教程中有说到，相比于以函数的方式定义的损失函数，类的方式定义更为常用。

> 虽然以函数定义的方式很简单，但是以类方式定义更加常用，…

然而，从教程中给出的例子，比如`DiceLoss`损失函数的定义

```python
class DiceLoss(nn.Module):
    def __init__(self,weight=None,size_average=True):
        super(DiceLoss,self).__init__()

    def forward(self,inputs,targets,smooth=1):
        inputs = F.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()                   
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        return 1 - dice

```

确实又难以体现出其相比函数方法的优越之处。考虑到类这种面向对象的设计方式，上述采用类的方式设计损失函数可能存在如下两个方面的优势：

- 当损失函数计算过程中出现一些类似滑动平均等需要动态缓存一些数的时候，采用类的方式可以直接将这样的数存放在实体对象中；
- 采用类的方式可以通过继承的方式梳理清楚不同损失函数的关系，并有可能能复用一些父类损失函数的特性和方法。
  

### 动态调整学习率

无论是在深度学习任务中还是深度强化学习任务中，学习率对于神经网络的训练非常重要。因为本质上讲，两者都是通过数据驱动的手段，通过梯度下降类算法，对神经网络的参数进行寻优。对于一个任务，在起始时，我们可能设定了一个比较好的学习率。这使得我们的算法在训练初期收敛的效率和效果都较好。但随着训练的进行，特别是当网络参数非常靠近我们期待的位置时（神经网络参数空间中的理想点），我们初期设置的学习率可能就会显得偏大，导致梯度下降过程步长过长，从而使得神经网络参数在理想点附近震荡。

为解决上述问题，一种方式是通过手动调整学习率，来适应神经网络训练不同的时期，以及神经网络所达到的不同性能。但这样的方式就要求我们要能够自行设计出一套学习率变化的算法，这无疑为我们程序训练的编写又增加了复杂度。另一种方式下，我们可以使用Pytorch中的scheduler进行动态的学习率调整。

Pytorch的scheduler可以提供两种使用方式的支持：官方提供的scheduler API和自定义的scheduler。

#### 官方提供的scheduler API

官方提供的scheduler API主要放在`torch.optim.lr_scheduler`中，具体包括



| scheduler API                                                | 说明                                                         | 参数                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `lr_scheduler.LambdaLR`                                      | 学习率`lr`为一个初始值乘以一个函数，当last_epoch=-1时，`lr`取值为初始值 | * optimizer (Optimizer) – Wrapped optimizer.<br/>* lr_lambda (function or list) – A function which computes a multiplicative factor given an integer parameter epoch, or a list of such functions, one for each group in optimizer.param_groups.<br/> |
| \* last_epoch (int) – The index of last epoch. Default: -1.  |                                                              |                                                              |
| \* verbose (bool) – If True, prints a message to stdout for each update. Default: False. |                                                              |                                                              |
| `lr_scheduler.MultiplicativeLR`                              | 学习率`lr`为一个初始值乘以一个函数，当last_epoch=-1时，`lr`取值为初始值 | * optimizer (Optimizer) – Wrapped optimizer.                 |
| * lr_lambda (function or list) – A function which computes a multiplicative factor given an integer parameter epoch, or a list of such functions, one for each group in optimizer.param_groups.<br/> |                                                              |                                                              |
| \* last_epoch (int) – The index of last epoch. Default: -1.  |                                                              |                                                              |
| \* verbose (bool) – If True, prints a message to stdout for each update. Default: False. |                                                              |                                                              |
| `lr_scheduler.StepLR`                                        | 每step_size个epoch，学习率`lr`变为其当前值乘以`gamma`，当last_epoch=-1时，`lr`取值为初始值 | * optimizer (Optimizer) – Wrapped optimizer.                 |
| \* step_size (int) – Period of learning rate decay.          |                                                              |                                                              |
| \* gamma (float) – Multiplicative factor of learning rate decay. Default: 0.1. |                                                              |                                                              |
| \* last_epoch (int) – The index of last epoch. Default: -1.  |                                                              |                                                              |
| \* verbose (bool) – If True, prints a message to stdout for each update. Default: False. |                                                              |                                                              |
| `lr_scheduler.MultiStepLR`                                   | 当epoch数达到milestones数量时，学习率`lr`变为其当前值乘以`gamma`，当last_epoch=-1时，`lr`取值为初始值 | * optimizer (Optimizer) – Wrapped optimizer.                 |
| * milestones (list) – List of epoch indices. Must be increasing. |                                                              |                                                              |
| * gamma (float) – Multiplicative factor of learning rate decay. / * Default: 0.1. |                                                              |                                                              |
| * last_epoch (int) – The index of last epoch. Default: -1.   |                                                              |                                                              |
| * verbose (bool) – If True, prints a message to stdout for each update. Default: False. |                                                              |                                                              |
| `lr_scheduler.ExponentialLR`                                 | 每个epoch，学习率`lr`变为其当前值乘以`gamma`，当last_epoch=-1时，`lr`取值为初始值 | * optimizer (Optimizer) – Wrapped optimizer.                 |
| * gamma (float) – Multiplicative factor of learning rate decay. |                                                              |                                                              |
| * last_epoch (int) – The index of last epoch. Default: -1.   |                                                              |                                                              |
| * verbose (bool) – If True, prints a message to stdout for each update. Default: False. |                                                              |                                                              |
| `lr_scheduler.CosineAnnealingLR`                             | 采用cos衰减的方式调整学习率，当last_epoch=-1时，`lr`取值为初始值 | * optimizer (Optimizer) – Wrapped optimizer.                 |
| * T_max (int) – Maximum number of iterations.                |                                                              |                                                              |
| * eta_min (float) – Minimum learning rate. Default: 0.       |                                                              |                                                              |
| * last_epoch (int) – The index of last epoch. Default: -1.   |                                                              |                                                              |
| * verbose (bool) – If True, prints a message to stdout for each update. Default: False. |                                                              |                                                              |
| `lr_scheduler.ReduceLROnPlateau`                             | 当某项指标不再下降时削减学习率                               | * ptimizer (Optimizer) – Wrapped optimizer.                  |
| * mode (str) – One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’. |                                                              |                                                              |
| * factor (float) – Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1. |                                                              |                                                              |
| * patience (int) – Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: 10. |                                                              |                                                              |
| * threshold (float) – Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4. |                                                              |                                                              |
| * threshold_mode (str) – One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: ‘rel’. |                                                              |                                                              |
| * cooldown (int) – Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0. |                                                              |                                                              |
| min_lr (float or list) – A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: 0. |                                                              |                                                              |
| * eps (float) – Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8. |                                                              |                                                              |
| * verbose (bool) – If True, prints a message to stdout for each update. Default: False. |                                                              |                                                              |
| `lr_scheduler.CyclicLR`                                      | 以某种循环策略调整学习率                                     | 详见https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR |
| `lr_scheduler.OneCycleLR`                                    | 以某种单次循环策略调整学习率                                 | 详见https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR |
| `lr_scheduler.CosineAnnealingWarmRestarts`                   | 采用cos衰减的方式调整学习率，当last_epoch=-1时，`lr`取值为初始值 |                                                              |



在训练中，上述scheduler API通过实例化创建scheduler实例，再通过**在optimizer优化一步（即调用step()方法）后**，调用step()方法进行学习率调整，如下：

```python
# 选择优化器
optimizer = torch.optim.Adam(...) 
# 选择一种或多种动态调整学习率方法
scheduler = torch.optim.lr_scheduler.... 

# 进行训练
for epoch in range(100):
    train(...)
    validate(...)
    optimizer.step()
    # 在优化器参数更新之后再动态调整学习率
    scheduler.step() 
    ...

```



#### 自定义scheduler

自定义scheduler的方法是通过构建自定义函数adjust_learning_rate，来改变optimizer的param_group中lr的值实现的，例如：

```python
def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

基于此定义的scheduler，我们便可以在训练时进行使用，如下：

```python
# 选择优化器
optimizer = torch.optim.Adam(...) 
# 进行训练
for epoch in range(100):
    train(...)
    validate(...)
    optimizer.step()
    # 调用自定义函数调整学习率
    adjust_learning_rate(optimizer, epoch)
    ...
```

#### 问题

在使用自定义学习率调整函数时，自定义学习率调整函数是否也要放在`optimizer.step()`语句之后？



### 模型微调

当前，模型参数的规模持续膨胀，能够达到的能力水平，甚至跨任务的泛化水平也在不断提高，但这些模型往往均是通过在大数据集上训练的。因此，当前在很多深度学习任务求解上的做法是基于一个在很大的数据集上训练的模型进行进一步调整实现的。这其实就是模型微调——基于预训练模型在当前任务上进行进一步训练。也正是基于这样的思想，近年来预训练大模型开始成为热点。从BERT到GPT-3，到大模型的出现一方面促进了AI模型泛化能力的提升，另一方面也削减了下游任务（具体任务）的训练成本，催生了“大模型预训练+微调”的应用研发范式。

Pytorch提供了许多预训练好的网络模型（VGG，ResNet系列，mobilenet系列…），这些模型都是PyTorch官方在相应的大型数据集训练好的。在面对具体下游任务时，我们可以从中选择与我们任务接近的模型，换成我们的数据进行精调，也就是我们经常见到的finetune。

#### 模型微调流程

模型精调分为如下几步：

1. 在源数据集上预训练一个神经网络模型，即源模型。这一步实际上预训练模型制作方为我们准备好了，即我们在Pytorch中拿到的就已经是预训练好的模型了。
2. 创建新的神经网络模型，即目标模型。将源模型中除了最终的输出层外所有部分的模型和相应的参数复制到目标模型中。这一步是通过模型结构和参数的拷贝，将源模型（预训练模型）预训练中的经验赋予目标模型。但由于目标模型与源模型面对的任务不同，因此，目标模型中最后的输出层保留独立。我认为，这里不仅限于输出层，扩充一些。只要是针对当前任务特有的层都可以保留相对于源模型的独立性。
3. 为目标模型添加一个与目标模型任务想匹配的输出层，并随机初始化该层的模型参数。
4. 在目标数据集上训练目标模型。对于输出层（即目标模型特有的部分），我们将从头训练，而其余层的参数都是基于源模型的参数微调得到的。
   

![image-20220908074048462](DeepLearning.assets/image-20220908074048462.png)





# 目标检测



[[PyTorch 学习笔记\] 8.2 目标检测简介 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/259494709)

在构造 DataLoader 时，还要传入一个`collate_fn()`函数。这是因为在目标检测中，图片的宽高可能不一样，无法以 4D 张量的形式拼接一个 batch 的图片，因此这里使用 tuple 来拼接数据。



```python
# 收集batch data的函数
    def collate_fn(batch):
        return tuple(zip(*batch))
```

collate_fn 的输入是 list，每个元素是 tuple；每个 tuple 是 Dataset 中的 `__getitem__()`返回的数据，包括`(image, target)`

举个例子：

```python
image=[1,2,3]
target=[4,5,6]
batch=list(zip(image,target))
print("batch:")
print(batch)
collate_result = tuple(zip(*batch))
print("collate_result:")
print(collate_result)
```

输出为：

```python
batch:
[(1, 4), (2, 5), (3, 6)]
collate_result:
((1, 2, 3), (4, 5, 6))
```





# python 基础



## 时间测试



[python 时间函数 毫秒_python 利用time模块给程序计时_小西老师的博客-CSDN博客](https://blog.csdn.net/weixin_30569303/article/details/112357544)



python的time内置模块是一个与时间相关的内置模块，很多人喜欢用time.time()获取当前时间的**时间戳**，利用程序前后两个时间戳的差值计算程序的运行时间，如下：

### 使用time.time()

```python
import time
T1 = time.time()
 
#______假设下面是程序部分______
for i in range(100*100):
    pass
 
T2 = time.time()
print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
# 程序运行时间:0.0毫秒
```

不要以为你的处理器很厉害，就忽视了一个问题，一万次遍历，时间为0.0毫秒？

下面解决上面的质疑

### 使用time.clock()

Python time clock() 函数以浮点数计算的秒数返回当前的CPU时间。用来衡量不同程序的耗时，比time.time()更有用。

这个需要注意，在不同的系统上含义不同。在UNIX系统上，它返回的是"进程时间"，它是用秒表示的浮点数（时间戳）。而在WINDOWS中，第一次调用，返回的是进程运行的实际时间。而第二次之后的调用是自第一次调用以后到现在的运行时间。（实际上是以WIN32上QueryPerformanceCounter()为基础，它比毫秒表示更为精确）

使用time.clock()更改后的程序查看一下：


```python
import platform
print('系统:',platform.system())
 
import time
T1 = time.clock()
 
#______假设下面是程序部分______
for i in range(100*100):
    pass
 
T2 =time.clock()
print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
# 程序运行时间:0.27023641716203606毫秒
```



### 使用time.perf_counter()

返回性能计数器的值（以微秒为单位,1秒=1000毫秒；1毫秒=1000微秒）作为浮点数，即具有最高可用分辨率的时钟，以测量短持续时间。它包括在睡眠期间和系统范围内流逝的时间。返回值的参考点未定义，因此只有连续调用结果之间的差异有效。

1秒 = 1000毫秒

1毫秒 = 1000微秒

1微秒 = 1000纳秒

1纳秒 = 1000皮秒

```python
import platform
print('系统:',platform.system())
 
import time
T1 = time.perf_counter()
 
#______假设下面是程序部分______
for i in range(100*100):
    pass
 
T2 =time.perf_counter()
print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
# 系统: Windows
# 程序运行时间:0.3007180604248629毫秒
```



### 使用time.process_time()

返回当前进程的系统和用户CPU时间总和的值（以小数微秒为单位）作为浮点数。

通常time.process_time()也用在测试代码时间上，根据定义，它在整个过程中。返回值的参考点未定义，因此我们测试代码的时候需要调用两次，做差值。

注意process_time()不包括sleep()休眠时间期间经过的时间。

```python
import platform
print('系统:',platform.system())
 
import time
T1 = time.process_time()
 
#______假设下面是程序部分______
for i in range(100*100):
    pass
 
T2 =time.process_time()
print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
# 系统: Windows
# 程序运行时间:0.0毫秒
```



建议PC上使用time.perf_counter() 来计算程序的运算时间，特别是测试算法在相邻两帧的处理时间，如果计算不准确，那可能会对算法的速度过于自信。

尤其在嵌入式的板子的开发中，性能的测试中，请仔细选择时间模块，比如某些嵌入式板子会封装专门的模块。



# 其他库使用



## SimpleCRF





## GeodisTK





Landmark-free Statistical Shape Modeling via Neural Flow Deformations

















