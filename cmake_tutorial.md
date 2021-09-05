#  cmake 学习笔记

可供学习的资源

[《CMake菜谱（CMake Cookbook中文版）》 - 书栈网 · BookStack](https://www.bookstack.cn/books/CMake-Cookbook)



CMake的find_package指令用于查找并载入一个外部包的设置。

## 基本调用形式和模块模式

```cmake
find_package(<PackageName> [version] [EXACT] [QUIET] [MODULE]
             [REQUIRED] [[COMPONENTS] [components...]]
             [OPTIONAL_COMPONENTS components...]
             [NO_POLICY_SCOPE])
```

