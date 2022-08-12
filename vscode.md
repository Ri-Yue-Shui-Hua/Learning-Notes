

# VSCODE使用

## 打开vscode



From the Developer Command Prompt, create an empty folder called "projects" where you can store all your VS Code projects, then create a subfolder called "helloworld", navigate into it, and open VS Code (`code`) in that folder (`.`) by entering the following commands:

在![image-20220723183857915](D:\GitHub\Learning-Notes\vscode.assets\image-20220723183857915.png) 中输入；

```bash
code
```

就可以打开vscode，并可以使用cl调试。

参考：[Configure Visual Studio Code for Microsoft C++](https://code.visualstudio.com/docs/cpp/config-msvc)



## debug vscode

其中Developer Command Prompt 先进入代码目录下，然后执行命令code，打开vscode，删除原task.json，重新生成task.json。

然后就可以调试了。编译器配置的是msvc。

![image-20220723202546119](D:\GitHub\Learning-Notes\vscode.assets\image-20220723202546119.png)