

CTK

[commontk/CTK: A set of common support code for medical imaging, surgical navigation, and related purposes. (github.com)](https://github.com/commontk/CTK)

## Common Toolkit

The Common Toolkit is a community effort to provide support code for medical image analysis, surgical navigation, and related projects.

See [http://commontk.org](http://commontk.org/)



## Build Instructions

Configure the project using CMake.

- For Qt5, specify the following:

  `CTK_QT_VERSION`: 5

  `QT5_DIR`: C:Qt5.15.0msvc2019_64libcmakeQt5 (or something similar, depending on operating system)

  `VTK_MODULE_ENABLE_VTK_GUISupportQt`: YES (for enabling VTK widgets)

  `VTK_MODULE_ENABLE_VTK_ViewsQt`: YES (for enabling VTK view widgets)

Note: make sure your built toolchain version is compatible with the chosen Qt version. For example if trying to build with Qt-5.12 and Microsoft Visual Studio 2019, then build will fail with the error error LNK2019: unresolved external symbol "__declspec(dllimport) public: __cdecl QLinkedListData::QLinkedListData(void)". The solution is to either change the toolset version to an earlier one (e.g., Visual Studio 2017) or upgrade Qt (e.g., use Qt-5.15 with Visual Studio 2019).







参考：

[myhhub/CTK-project: CTK完整教程(OSGI for C++ 实现 C++ Qt 模块化)。本教程围绕 CTK Plugin Framework，探索 C++ 中的模块化技术，并能够基于 CTK 快速搭建 C++ 组件化框架，避免后来的人走弯路。 (github.com)](https://github.com/myhhub/CTK-project)

[CTK编译教程(64位环境 Windows + Qt + MinGW或MSVC + CMake) | 来唧唧歪歪(Ljjyy.com) - 多读书多实践，勤思考善领悟](https://www.ljjyy.com/archives/2021/02/100643.html)

[CTK-使用ctk框架完成日志、打印、界面插件-C/C++ (uml.org.cn)](http://www.uml.org.cn/c++/202104072.asp?artid=23827)

[CTK框架——CTK Widgets快速入门_51CTO博客_ctk框架](https://blog.51cto.com/quantfabric/2120383)















