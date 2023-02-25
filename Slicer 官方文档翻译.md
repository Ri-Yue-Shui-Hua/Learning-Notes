# Slicer 官方文档翻译



# python脚本



## 从这里开始

请阅读[这些幻灯片](https://docs.google.com/presentation/d/1JXIfs0rAM7DwZAho57Jqz14MRn2BIMrjB17Uj_7Yztc/edit?usp=sharing)。这将使您对可能的操作有一个总体的了解，您可以使用二进制下载的Slicer来完成示例代码。

请参考包[含所有文档链接的描述](http://www.na-mic.org/Wiki/index.php/2013_Project_Week_Breakout_Session:Slicer4Python)。

查看[脚本存储库](https://slicer.readthedocs.io/en/latest/developer_guide/script_repository.html)以获得示例和灵感。



## 背景

这是[slicer3中的python实现](https://www.slicer.org/wiki/Slicer3:Python)的进化。Slicer的api现在原生地封装在python中。

在slicer4中，绘图等主题仍处于试验阶段。

请参阅[2012年关于slicer4中python状态的演示](http://www.na-mic.org/Wiki/index.php/AHM2012-Slicer-Python)。

有关更多示例，请参阅[python slicer4教程](https://www.slicer.org/wiki/Documentation/4.10/Training#Slicer4_Programming_Tutorial)。

[Slicer Self Tests](https://www.slicer.org/wiki/Documentation/Nightly/Developers/Tutorials/SelfTestModule)可以用python编写，并为操作Data、Logic和Slicer的gui提供了很好的示例来源。

## 从这里开始脚本化模块和扩展开发

为[NA-MIC 2013夏季项目周的Slicer/Python分组会议](http://www.na-mic.org/Wiki/index.php/2013_Project_Week_Breakout_Session:Slicer4Python)创建了一个广泛的教程和参考页面.

## 使用选项

Slicer是一个嵌入Python解释器的应用程序，类似于[Blender](https://docs.blender.org/api/blender_python_api_current/info_overview.html), [FreeCAD](https://www.freecadweb.org/wiki/Embedding_FreeCAD), [Calibre](https://manual.calibre-ebook.com/develop.html#using-an-interactive-python-interpreter), [Cinema4D](https://developers.maxon.net/docs/Cinema4DPythonSDK/html/manuals/introduction/python_in_c4d.html)等。

虽然Slicer有一组核心库，可以以某种方式打包，以便它们可以在任何Python环境中导入(import slicer)，但目前我们不提供这个选项。对于大多数其他应用程序来说，这也只是一个有限的实验性选项。

我们更愿意将精力集中在其他类型的Python集成上:

- 官方推荐的运行Python处理脚本的方式是使用Slicer的嵌入式Python解释器(例如，Python交互器)执行它们。任何其他所需的python包都可以使用pip (pip_install('scipy')安装。
- Python调试器(PyCharm, VS Code, Eclipse等)可以附加到正在运行的Slicer应用程序。
- SlicerJupyter扩展使Slicer应用程序充当Jupyter内核，同时保持应用程序完全交互。
- Slicer可以启动任何外部Python脚本(GUI是从XML描述符自动生成的)，默认情况下在后台处理线程中运行，而不会阻塞GUI。请看这里的[例子](https://github.com/lassoan/SlicerPythonCLIExample)。



## Python Interactor(交互器)

使用Window->Python交互器(在Window /linux上是`Control-3`，在mac上是`Command-3`)来打开基于Qt的控制台，可以访问vtk、Qt和Slicer包装的api。大多数python代码都可以从这个窗口安装和运行，但因为它存在于事件驱动的Qt GUI环境中，所以不容易支持一些操作，比如并行处理或无头操作。

### Examples (例子)

启动Slicer，打开python控制台。像这样加载一个样本volume:

```python
import SampleData
sampleDataLogic = SampleData.SampleDataLogic()
sampleDataLogic.downloadMRHead()
```

获取该volume的volumeNode:

```python
n = getNode('MRHead')
```

可以使用Tab查看类实例的方法列表。

### 访问 Volume data as numpy array

您可以使用numpy和相关代码轻松地检查和操作volume数据。在Slicer中你可以这样做:

```python
a = array('MRHead')
```

'a'将是一个指向适当数据的指针(没有数据复制)。如果你得到一个“array”没有定义的error，那么运行“import slicer.util”。然后使用'slice . util.array'。Scalar volumes变成三维阵列，vector volumes变成4D，tensor volumes变成5D。所有数组都可以直接操作。修改完成后，调用volumeNode的Modified()方法表示图像被修改并触发显示更新。

**array**方法仅用于快速测试，因为多个节点可能具有相同的名称，并且可以从MRML节点恢复各种数组。在Slicer模块中，建议改用**arrayFromVolume**，它接受MRML节点作为输入。

```python
volumeNode = getNode('MRHead')
a = arrayFromVolume(volumeNode)
# Increase image contrast
a[:] = a * 2.0
arrayFromVolumeModified(volumeNode)
```

如果你没有就地(in-place)处理数据，但你在numpy数组中有计算结果，那么你必须使用**updateVolumeFromArray**将numpy数组的内容复制到volume中:

```python
import numpy as np
import math

def some_func(x, y, z):
  return 0.5*x*x + 0.3*y*y + 0.5*z*z

a = np.fromfunction(some_func,(30,20,15))

volumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
volumeNode.CreateDefaultDisplayNodes()
updateVolumeFromArray(volumeNode, a)
setSliceViewerLayers(background=volumeNode)
```













参考 [Documentation/Nightly/Developers/Python scripting - Slicer Wiki](https://www.slicer.org/wiki/Documentation/Nightly/Developers/Python_scripting)