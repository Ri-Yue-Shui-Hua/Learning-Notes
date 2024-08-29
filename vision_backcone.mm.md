
# 通用Vision BackBone


## 1. 视觉MLP首创：MLP-Mixer
### 1. MLP-Mixer: An all-MLP Architecture for Vision
- Google Research, Brain Team，ViT 作者团队

### 2. RepMLP：卷积重参数化为全连接层进行图像识别
- 清华大学，旷视，RepVGG 作者团队

### 3. ResMLP：ImageNet 数据集训练残差 MLP 网络
-  Facebook AI，索邦大学

## 2. 视觉MLP进阶方法
### 4. 谷歌大脑提出gMLP:请多关注MLP
- 谷歌大脑，Quoc V .Le 团队

### 5. 港大提出 CycleMLP：用于密集预测的类似 MLP 的架构
- 港大，罗平教授团队

## 3. 傅里叶变换的类 MLP 架构 (1)
### 6. GFNet：将 FFT 思想用于空间信息交互

## 4. 匹敌 Transformer 的2020年代的卷积网络
### 7. ConvNeXt
- FAIR，UCB

## 5. 傅里叶变换的类 MLP 架构 (2)
### 8. AFNO：自适应傅里叶神经算子
- NVIDIA，加州理工，斯坦福大学

## 6. 图神经网络打造的通用视觉架构

### 9. Vision GNN：把一张图片建模为一个图
- 中国科学院大学，华为诺亚方舟实验室，北大


## 7. 优化器的重参数化技术
### 10. RepOptimizer：重参数化你的优化器：VGG 型架构 + 特定的优化器 = 快速模型训练 + 强悍性能
- 清华大学，旷视科技，RepVGG 作者工作

## 8. 递归门控卷积打造的通用视觉架构
### 11. HorNet：通过递归门控卷积实现高效高阶的空间信息交互
- 清华大学，周杰，鲁继文团队，Meta AI

## 9. 用于通用视觉架构的 MetaFormer 基线
### 12. MetaFormer：令牌混合器类型不重要，宏观架构才是通用视觉模型真正需要的
- Sea AI Lab，新加坡国立大学

## 10. 将卷积核扩展到 51×51
### 13. SLaK：从稀疏性的角度将卷积核扩展到 51×51
- 埃因霍温理工大学，德州农工

## 11. Transformer 风格的卷积网络视觉基线模型
### 14. Conv2Former：Transformer 风格的卷积网络视觉基线模型
- 南开大学，字节跳动

## 12. 无注意力机制视觉 Transformer 的自适应权重混合
### 15. AMixer：无注意力机制视觉 Transformer 的自适应权重混合
- 清华大学

## 13. 简单聚类算法实现强悍视觉架构
### 16. 把图片视为点集，简单聚类算法实现强悍视觉架构 (ICLR 2023 超高分论文)

## 14. 2020年代的卷积网络适配自监督学习
### 17. ConvNeXt V2：使用 MAE 协同设计和扩展 ConvNets
- KAIST，Meta AI，FAIR，纽约大学 [ConvNeXt 原作者刘壮，谢赛宁团队]

## 15. 一个适应所有 Patch 大小的 ViT 模型
### 18.  FlexiViT：一个适应所有 Patch 大小的 ViT 模型
- 谷歌，ViT，MLP-Mixer 作者团队

## 16. 空间 Shift 操作实现通用基础视觉 MLP
### 19. S^2-MLP：空间 Shift 操作实现通用基础视觉 MLP
- 百度

## 17. Base Model 训练策略的研究
### 20. ResNet 的反击：全新训练策略带来强悍 ResNet 性能
- timm 作者，DeiT 一作

## 18. 首个适用下游任务的轴向移位 MLP
### AS-MLP：首个适用下游任务的轴向移位 MLP 视觉骨干架构
- 上海科技大学

## 19. 当移位操作遇到视觉 Transformer
### 22. ShiftViT：当移位操作遇到视觉 Transformer
- 中国科学技术大学，MSRA

## 20. 用于密集预测任务的视觉 Transformer Adapter
### 23. ViT-Adapter：用于密集预测任务的视觉 Transformer Adapter
- 南大，Shanghai AI Lab，清华

## 21. ViT 的前奏：Scale up 卷积神经网络学习通用视觉表示
### 24. ViT 的前奏：Scale up 卷积神经网络学习通用视觉表示
- 谷歌，含 ViT 作者团队

## 22. FasterNet：追求更快的神经网络
### 25. FasterNet：追求更快的神经网络
- HKUST，Rutgers University

## 23. AFFNet：频域自适应频段过滤=空域全局动态大卷积核
### 26. AFFNet：频域自适应频段过滤=空域全局动态大卷积核
-  MSRA

## 24. Flattened Transformer：聚焦的线性注意力机制构建视觉 Transformer
### 27. Flatten Transformer：聚焦的线性注意力机制构建视觉 Transformer
- 清华，黄高老师团队

## 25. RefConv：一种基于重参数化操作的重聚焦卷积方法
### 28. RefConv：一种基于重参数化操作的重聚焦卷积方法
- 自南京大学，腾讯 AI Lab，RepVGG 作者团队

## 26. NFNet：无需 BN 的 ResNet 变体
### 29. NFNet：无需 BN 的 ResNet 变体
- DeepMind

## 27. NFNet 视觉大模型：匹敌 ViT 性能的 JFT-4B 大规模预训练
### 30. NFNet 视觉大模型：匹敌 ViT 性能的大规模预训练
- Google DeepMind


## 28. 在 ViT 中使用 ReLU 取代 Softmax
### 31. 在 ViT 中使用 ReLU 取代 Softmax
- Google DeepMind

## 29. 视觉 Transformer 需要寄存器
### 32. 视觉 Transformer 需要寄存器
- FAIR, Meta
  
## 30. Agent Attention：集成 Softmax 和 Linear 注意力机制
### 33. Agent Attention：集成 Softmax 和 Linear 注意力机制
- 清华，黄高老师团队

## 31. 一个像素就是一个 token！探索 Transformer 新范式
### 34. 一个像素就是一个 token！探索 Transformer 新范式
- FAIR, Meta AI，阿姆斯特丹大学
