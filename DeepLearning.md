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













