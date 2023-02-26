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

#### 访问 Volume data as numpy array

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

#### 将Model数据作为numpy数组访问

通过调用' arrayFromModelPoints '，您可以使用numpy和相关代码轻松地检查和操作model的点坐标。修改完成后，调用polydata上的Modified()方法，以指示model已被修改并触发显示更新。

```python
# Create a model containing a sphere
sphere = vtk.vtkSphereSource()
sphere.SetRadius(30.0)
sphere.Update()
modelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
modelNode.SetAndObservePolyData(sphere.GetOutput())
modelNode.CreateDefaultDisplayNodes()
a = arrayFromModelPoints(modelNode)
# change Y scaling
a[:,2] = a[:,2] * 2.5
arrayFromModelPointsModified(modelNode)
```

#### 从Python运行CLI

[本文档部分已转移到ReadTheDocs](https://slicer.readthedocs.io/en/latest/developer_guide/python_faq.html#how-to-run-a-cli-module-from-python)。

##### 从切片视图访问Slicer的vtkRenderWindows

下面的例子展示了如何获得渲染的切片窗口。

```python
lm = slicer.app.layoutManager()
redWidget = lm.sliceWidget('Red')
redView = redWidget.sliceView()
wti = vtk.vtkWindowToImageFilter()
wti.SetInput(redView.renderWindow())
wti.Update()
v = vtk.vtkImageViewer()
v.SetColorWindow(255)
v.SetColorLevel(128)
v.SetInputConnection(wti.GetOutputPort())
v.Render()
```

## 脚本库(Script Repository)

有关更大的示例代码集合，请参阅[脚本存储库](https://slicer.readthedocs.io/en/latest/developer_guide/script_repository.html)。



## 开发人员常见问题:Python脚本

### How to run pip ?

pip可执行文件不是分布式的，而是应该使用以下命令:

- from build tree: `/path/to/Slicer-SuperBuild/python-install/bin/PythonSlicer -m pip ...`
- from install tree:

```bash
* Linux/MacOS: /path/to/Slicer-X.Y.Z-plat-arch/bin/PythonSlicer -m pip ...
 * Windows: "c:\Program Files\Slicer 4.10.0\bin\PythonSlicer.exe" -m pip ...
```

有关更多细节和背景信息，请参阅此讨论:https://discourse.slicer.org/t/slicer-python-packages-use-and-install/984/29

### 如何从python脚本访问脚本模块

所有Slicer模块都可以在slicer.modules命名空间中访问。例如，sampledata模块可以通过slicer.modules.sampledata访问。

要访问模块的小部件（widget），请使用widgetRepresentation()方法获取c++基类，并使用它的self()方法获取Python类。例如，slice .modules.sampledata. widgetRepresentation ().self()返回sampledata模块的Python小部件(widget)对象。

### 如何在启动时系统地执行自定义python代码?

参看[https://slicer.readthedocs.io/en/latest/user_guide/settings.html application-startup-file](https://slicer.readthedocs.io/en/latest/user_guide/settings.html#application-startup-file)

### 如何使用python保存一个Image/Volume ?

模块slicer.util提供了允许保存节点或整个场景的方法:

- saveNode
- saveScene

详情见:

- https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/util.py#L229-267
- https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/tests/test_slicer_util_save.py

#### 在保存volume时启用或禁用压缩

虽然可以在Slicer Python模块中以vtkMRMLVolumeNode的形式访问volume，但应该将压缩首选项(或与此相关的任何其他属性)传递给Slicer .util. saveNode函数。该属性将被传递到Slicer的存储节点。对于压缩，将useCompression设置为0或1。示例脚本:

```python
properties = {'useCompression': 0}; #do not compress
file_path = os.path.join(case_dir, file_name)
slicer.util.saveNode(node, file_path, properties)
```

### 如何将volume分配给Slice视图?

假设MRHead示例数据已经加载，您可以执行以下操作:

```python
red_logic = slicer.app.layoutManager().sliceWidget("Red").sliceLogic()
red_cn = red_logic.GetSliceCompositeNode()
red_logic.GetSliceCompositeNode().SetBackgroundVolumeID(slicer.util.getNode('MRHead').GetID())
```

### 如何在Slicer 3D视图中访问vtkRenderer ?

```python
renderer = slicer.app.layoutManager().threeDWidget(0).threeDView().renderWindow().GetRenderers().GetFirstRenderer()
```

### 如何获得VTK渲染后端?

```python
backend = slicer.app.layoutManager().threeDWidget(0).threeDView().renderWindow().GetRenderingBackend()
```

### 如何访问与Slicer 2D或3D视图相关的可显示管理器?

正如最初在[这里](http://slicer-devel.65872.n3.nabble.com/How-to-get-the-point-of-a-3D-model-based-on-the-fiducial-position-td4031760.html#a4031762)解释的那样，您可以使用任何[qMRMLThreeDView](http://slicer.org/doc/html/classqMRMLThreeDView.html)和[qMRMLSliceView](http://slicer.org/doc/html/classqMRMLSliceView.html)中可用的`getDisplayableManagers()`方法。

```python
lm = slicer.app.layoutManager()
for v in range(lm.threeDViewCount):
  td = lm.threeDWidget(v)
  ms = vtk.vtkCollection()
  td.getDisplayableManagers(ms)
  for i in range(ms.GetNumberOfItems()):
   m = ms.GetItemAsObject(i)
   if m.GetClassName() == "vtkMRMLModelDisplayableManager":
     print(m)
```

### 如何在场景中居中3D视图?

```python
layoutManager = slicer.app.layoutManager()
threeDWidget = layoutManager.threeDWidget(0)
threeDView = threeDWidget.threeDView()
threeDView.resetFocalPoint()
```

### 我应该在脚本模块中使用“old style”还是“new style”python类?

当python类没有指定超类时，它们是“旧风格”，如这里[[1]](http://docs.python.org/2/reference/datamodel.html#new-style-and-classic-classes)所述

一般来说，脚本模块中的类并不重要，因为它们不会被子类化，旧的或新的样式应该是相同的。

对于slicer中可能要子类化的其他python代码，最好使用新样式类。有关示例，请参阅[EditorLib](https://github.com/Slicer/Slicer/tree/master/Modules/Scripted/EditorLib)和[DICOMLib](https://github.com/Slicer/Slicer/tree/master/Modules/Scripted/DICOM/DICOMLib)中的类层次结构。

### 如何强化变换（harden a transform） ?

```python
>>> n = getNode('Bone')
>>> logic = slicer.vtkSlicerTransformLogic()
>>> logic.hardenTransform(n)
```

讨论:http://slicer-devel.65872.n3.nabble.com/Scripting-hardened-transforms-tt4029456.html

### 我可以在哪里找到示例脚本?

看一下脚本存储库([Script repository](https://slicer.readthedocs.io/en/latest/developer_guide/script_repository.html))。

### 如何使用可视化调试器进行分步调试

使用各种IDE可以实现Python脚本模块的可视化调试(设置断点、逐步执行代码、查看变量、堆栈等)。请在[这里](https://slicer.readthedocs.io/en/latest/developer_guide/debugging/overview.html#python-debugging)查看详细的安装说明。

### 为什么我不能从python访问我的c++ Qt类

- Qt类的Python包装需要一个以QObject作为参数的Qt样式构造函数(尽管它可以默认为null)，它是公共的。如果缺少其中一个，该类的python包装将失败。
- 在实例化的python类的作用域之外，不能从python访问自定义c++ Qt类。这些都不起作用:

```python
BarIDRole = slicer.qFooItemDelegate.LastRole + 1

class BarTableWidget(qt.QTableWidget, VTKObservationMixin):

    def __init__(self, *args, **kwargs):
        [...]
```



```python
class BarTableWidget(qt.QTableWidget, VTKObservationMixin):

    BarIDRole = slicer.qFooItemDelegate.LastRole + 1

    def __init__(self, *args, **kwargs):
        [...]
```

反之，取：

```python
class BarTableWidget(qt.QTableWidget, VTKObservationMixin):

    def __init__(self, *args, **kwargs):
        self.BarIDRole = slicer.qFooItemDelegate.LastRole + 1
        [...]
```

- [Other reasons go here]

### 我可以使用工厂方法，如CreateNodeByClass或GetNodesByClass吗?

看 https://slicer.readthedocs.io/en/latest/developer_guide/advanced_topics.html#memory-management

### 如何访问一个VTK对象观察者回调函数中的callData参数

要获得有关由VTK对象发出的事件的通知，您可以简单地使用AddObserver方法，例如:

```python
def sceneModifiedCallback(caller, eventId):
  print("Scene modified")
  print("There are {0} nodes in the scene". format(slicer.mrmlScene.GetNumberOfNodes()))

sceneModifiedObserverTag = slicer.mrmlScene.AddObserver(vtk.vtkCommand.ModifiedEvent, sceneModifiedCallback)
```

如果一个事件还包含CallData这样的附加信息，那么这个参数的类型也必须指定，例如:

```python
@vtk.calldata_type(vtk.VTK_OBJECT)
def nodeAddedCallback(caller, eventId, callData):
  print("Node added")
  print("New node: {0}".format(callData.GetName()))

nodeAddedModifiedObserverTag = slicer.mrmlScene.AddObserver(slicer.vtkMRMLScene.NodeAddedEvent, nodeAddedCallback)
```

**注意**:@vtk.calldata_type是一个Python装饰器，它修改在装饰器之后声明的函数的属性。装饰器在VTK中定义(在[Wrapping\Python\vtk\util\misc.py](https://github.com/Kitware/VTK/blob/master/Wrapping/Python/vtkmodules/util/misc.py)中)。

**注意**: 可用的类型在列出如 [Wrapping\Python\vtkmodules\util\vtkConstants.py](https://github.com/Kitware/VTK/blob/master/Wrapping/Python/vtkmodules/util/vtkConstants.py).

从类中使用需要在类__init__函数中创建回调的额外步骤，因为Python2默认情况下会做一些额外的包装 (http://stackoverflow.com/questions/9523370/adding-attributes-to-instance-methods-in-python):

```python
class MyClass:
  def __init__(self):
    from functools import partial
    def nodeAddedCallback(self, caller, eventId, callData):
      print("Node added")
      print("New node: {0}".format(callData.GetName()))
    self.nodeAddedCallback = partial(nodeAddedCallback, self)
    self.nodeAddedCallback.CallDataType = vtk.VTK_OBJECT
  def registerCallbacks(self):
    self.nodeAddedModifiedObserverTag = slicer.mrmlScene.AddObserver(slicer.vtkMRMLScene.NodeAddedEvent, self.nodeAddedCallback)
  def unregisterCallbacks(self):
    slicer.mrmlScene.RemoveObserver(self.nodeAddedModifiedObserverTag)
        
myObject = MyClass()
myObject.registerCallbacks()
```

允许的CallDataType值:VTK_STRING、VTK_OBJECT、VTK_INT、VTK_LONG、VTK_DOUBLE、VTK_FLOAT、"string0"。点击这里查看更多信息:

https://github.com/Kitware/VTK/blob/master/Wrapping/PythonCore/vtkPythonCommand.cxx

有关简化的语法，请参阅[#How_to_manage_VTK_object_connections_.3F](https://www.slicer.org/wiki/Documentation/Nightly/Developers/Python_scripting#How_to_manage_VTK_object_connections_.3F)



### 如何管理VTK对象连接?

VTKObservationMixin是一个Python混合程序，允许通过继承向类添加一组方法。它包括在[Base/Python/slicer/util.py](https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/util.py#L995)中定义的以下方法:

- addObserver 
- hasObserver 
- observer 
- removeObserver 
- removeObservers

下面演示了如何观察 `slicer.vtkMRMLScene.NodeAddedEvent`:

```python
from slicer.util import VTKObservationMixin

class MyClass(VTKObservationMixin):
  def __init__(self):
    VTKObservationMixin.__init__(self)
    self.addObserver(slicer.mrmlScene, slicer.vtkMRMLScene.NodeAddedEvent, self.nodeAddedCallback)
  
  @vtk.calldata_type(vtk.VTK_OBJECT)
  def nodeAddedCallback(self, caller, eventId, callData):
    print("Node added")
    print("New node: {0}".format(callData.GetName()))

myObject = MyClass()
```

有关使用的其他示例，请参见:  [test_slicer_util_VTKObservationMixin.py](https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/tests/test_slicer_util_VTKObservationMixin.py)

### 如果我试图访问数组中不存在的项，Slicer将崩溃

例如，这段代码使Slicer崩溃:

```python
s = vtk.vtkStringArray()
s.GetValue(0)
```

这种行为是预期的，因为所有的VTK对象都是用c++实现的，提供了更快的操作，但开发人员必须注意只寻址有效的数组元素，例如通过检查数组中元素的数量:

```python
if itemIndex < 0 or itemIndex >= s.GetNumberOfValues()
  raise IndexError("index out of bounds")
```

### 如何从Python运行CLI模块?

看[这里](https://www.slicer.org/wiki/Documentation/Nightly/Developers/Python_scripting#Running_a_CLI_from_Python)。

### 如何从批处理脚本运行Slicer操作?

```python
Slicer --no-main-window --python-script /tmp/test.py
```

/tmp/test.py内容：

```python
# use a slicer scripted module logic
from SampleData import SampleDataLogic
SampleDataLogic().downloadMRHead()
head = slicer.util.getNode('MRHead')

# use a vtk class
threshold = vtk.vtkImageThreshold()
threshold.SetInputData(head.GetImageData())
threshold.ThresholdBetween(100, 200)
threshold.SetInValue(255)
threshold.SetOutValue(0)

#  use a slicer-specific C++ class
erode = slicer.vtkImageErode()
erode.SetInputConnection(threshold.GetOutputPort())
erode.SetNeighborTo4()  
erode.Update()          

head.SetAndObserveImageData(erode.GetOutputDataObject(0))

slicer.util.saveNode(head, "/tmp/eroded.nrrd")

exit()
```

### 如何在无头计算节点上运行Slicer?

许多集群节点都安装了不包括X服务器的最小linux系统。X服务器，特别是那些具有硬件加速的服务器，传统上需要安装root权限，这使得无法运行使用X或OpenGL呈现的应用程序。

但是有一个解决办法，它允许Slicer中的一切正常工作，所以你甚至可以做无头渲染。

您可以使用支持运行虚拟framebuffer的X的现代版本。这可以在用户模式下安装，因此您甚至不需要在系统上拥有root。

详见[[2]](https://www.xpra.org/trac/wiki/Xdummy)。

这里有一个讨论更多的帖子:[[3]](http://massmail.spl.harvard.edu/public-archives/slicer-devel/2015/017317.html)

下面是在运行CTK测试(也使用Qt和VTK)的无头计算节点上运行的方法的工作示例 [[4]](https://github.com/pieper/CTK/blob/master/.travis.yml)

### 如何保存用户在场景中选择的参数和节点?

最好将用户在用户界面上所做的所有参数值和节点选择保存到MRML场景中。这允许用户加载一个场景，并从他离开的地方继续。这些信息可以保存在一个 *slicer.vtkMRMLScriptedModuleNode()*节点中。

例如:

```python
parameterNode=slicer.vtkMRMLScriptedModuleNode()

# Save parameter values and node references to parameter node

alpha = 5.0
beta = "abc"
inputNode = slicer.util.getNode("InputNode")

parameterNode.SetParameter("Alpha",str(alpha))
parameterNode.SetParameter("Beta", beta)
parameterNode.SetNodeReferenceID("InputNode", inputNode.GetID())

# Retrieve parameter values and node references from parameter node

alpha = float(parameterNode.GetParameter("Alpha"))
beta = parameterNode.GetParameter("Beta")
inputNode = parameterNode.GetNodeReference("InputNode")
```

脚本模块的逻辑类有一个辅助函数getParameterNode，它返回一个参数节点，该参数节点对于特定模块是唯一的。如果尚未创建参数节点，则该函数将创建参数节点。默认情况下，参数节点是一个单例节点，这意味着场景中只有该节点的一个实例。如果最好允许参数节点的多个实例，则将逻辑对象的isSingletonParameterNode成员设置为False。

### 如何加载一个UI文件?

详见 [Documentation/Nightly/Developers/Tutorials/PythonAndUIFile](https://www.slicer.org/wiki/Documentation/Nightly/Developers/Tutorials/PythonAndUIFile)

### 如何更新进度条在脚本(Python，或其他)CLI模块

正如 [Execution Model documentation](https://www.slicer.org/wiki/Documentation/Nightly/Developers/SlicerExecutionModel#Showing_Progress_in_an_Application|Slicer)中详细描述的那样，Slicer解析打印在stdout上的特定格式的XML命令，以允许任何进程外的CLI程序向主Slicer应用程序报告进度(这将导致进度条更新)。但是，非常重要的是要注意在每个print语句之后必须刷新输出，否则Slicer在进程结束之前不会解析进度部分。参见调用

```python
sys.stdout.flush()
```

在如下所示的Python CLI示例中:

```python
#!/usr/bin/env python-real

if __name__ == '__main__':
  import time
  import sys
  
  print("""<filter-start><filter-name>TestFilter</filter-name><filter-comment>ibid</filter-comment></filter-start>""")
  sys.stdout.flush()

  for i in range(0,10):
      print("""<filter-progress>{}</filter-progress>""".format(i/10.0))
      sys.stdout.flush()
      time.sleep(0.5)

  print("""<filter-end><filter-name>TestFilter</filter-name><filter-time>10</filter-time></filter-end>""")
  sys.stdout.flush()
```

### 如何在脚本模块中显示CLI模块执行的进度条

```python
def createProgressDialog(parent=None, value=0, maximum=100, windowTitle="Processing..."):
    import qt # qt.qVersion()
    progressIndicator = qt.QProgressDialog()  #(parent if parent else self.mainWindow())
    progressIndicator.minimumDuration = 0
    progressIndicator.maximum = maximum
    progressIndicator.value = value
    progressIndicator.windowTitle = windowTitle
    return progressIndicator
      

class MyModuleWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def setup(self)
        parametersFormLayout = qt.QFormLayout(myInputCollapsibleButton)
        self.testButton = qt.QPushButton('Test')
        self.testButton.enabled = True
        self.testButton.clicked.connect(self.onTestButton)
        parametersFormLayout.addRow(self.TestButton)
    
    def onTestButton(self):
        myCli = slicer.modules.tmp2cli
        cliNode = None
        myInt = 100
        cliNode = slicer.cli.run(myCli, cliNode, {'myString': 'hello World', 'myInt':100} )
        cliNode.AddObserver('ModifiedEvent', self.onCliModified)
        self.progressBar = myUtil.createProgressDialog(None, 0, myInt)
    
    def onCliModified(self, caller, event):
        self.progressBar.setValue(caller.GetProgress())
        if caller.GetStatus() == slicer.vtkMRMLCommandLineModuleNode.Completed: 
            self.progressBar.close()
```

参见完整的[示例](https://discourse.slicer.org/t/how-to-update-the-progress-bar-from-a-scripted-cli/3789/2?u=lassoan)。

### 如何使用非Slicer Python环境运行Python脚本

如果你需要使用不同的环境(Python3, Anaconda等)而不是Slicer的嵌入式解释器运行Python脚本，那么你需要使用默认的启动环境运行Python可执行文件。从Slicer的环境启动将导致加载Slicer的Python库，这些库预计与外部环境的二进制不兼容，因此将导致外部应用程序崩溃。

在Linux下使用系统Python3从Slicer运行python代码的例子:

```python
command_line = ["/usr/bin/python3", "-c", "print('hola')"]
from subprocess import check_output
command_result = check_output(command_line, env=slicer.util.startupEnvironment())
print(command_result)
```

这个示例脚本只输出'hola'并退出，但代码可以被更复杂的指令或启动Python脚本所取代。

使用Anaconda运行python代码的例子，在Windows上使用名为“workspace-gpu”的环境:

```python
command_line = [r"c:\Users\msliv\Miniconda3\envs\workspace-gpu\python.exe", "-c", "print('hola')"]
from subprocess import check_output
command_result = check_output(command_line, env=slicer.util.startupEnvironment())
print(command_result)
```

### 如何配置网络代理?

当使用`urllib.request`或 `requests`，通过设置http_proxy和https_proxy环境变量配置的代理将被自动使用。

详情如下:

- https://docs.python.org/3/library/urllib.request.html#urllib.request.urlopen
- https://2.python-requests.org/en/master/user/advanced/#proxies



参考 [Documentation/Nightly/Developers/Python scripting - Slicer Wiki](https://www.slicer.org/wiki/Documentation/Nightly/Developers/Python_scripting)