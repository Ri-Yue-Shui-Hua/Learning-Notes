# CMake入门

## 什么是CMake

你或许听过好几种 Make 工具，例如 GNU Make，QT 的 qmake，微软的 MS nmake，BSD Make pmake，[Makepp](http://makepp.sourceforge.net/)，等等。这些 Make 工具遵循着不同的规范和标准，所执行的 Makefile 格式也千差万别。这样就带来了一个严峻的问题：如果软件想跨平台，必须要保证能够在不同平台编译。而如果使用上面的 Make 工具，就得为每一种标准写一次 Makefile ，这将是一件让人抓狂的工作。

CMake 就是针对上面问题所设计的工具：它首先允许开发者编写一种平台无关的 CMakeList.txt 文件来定制整个编译流程，然后再根据目标用户的平台进一步生成所需的本地化 Makefile 和工程文件，如 Unix 的 Makefile 或 Windows 的 Visual Studio 工程。从而做到“Write once, run everywhere”。显然，CMake 是一个比上述几种 make 更高级的编译配置工具。一些使用 CMake 作为项目架构系统的知名开源项目有 [VTK](http://www.vtk.org/)、[ITK](http://www.itk.org/)、[KDE](http://kde.org/)、[OpenCV](http://www.opencv.org.cn/opencvdoc/2.3.2/html/modules/core/doc/intro.html)、[OSG](http://www.openscenegraph.org/) 等 [[1\]](https://www.hahack.com/codes/cmake/#fn1)。



在 linux 平台下使用 CMake 生成 Makefile 并编译的流程如下：

1. 编写 CMake 配置文件 CMakeLists.txt 。
2. 执行命令 `cmake PATH` 或者 `ccmake PATH` 生成 Makefile（`ccmake` 和 `cmake` 的区别在于前者提供了一个交互式的界面）。其中， `PATH` 是 CMakeLists.txt 所在的目录。
3. 使用 `make` 命令进行编译。



本文将从实例入手，一步步讲解 CMake 的常见用法，文中所有的实例代码可以在[这里](https://github.com/myhhub/cmake-project)找到。如果你读完仍觉得意犹未尽，可以继续学习我在文章末尾提供的其他资源。



### 入门案例：单个源文件

本节对应的源代码所在目录：[Demo1](https://github.com/myhhub/cmake-project/tree/master/Demo1)。

对于简单的项目，只需要写几行代码就可以了。例如，假设现在我们的项目中只有一个源文件 [main.cc](https://github.com/myhhub/cmake-project/tree/master/Demo1/main.cc/) ，该程序的用途是计算一个数的指数幂。

```c++
#include <stdio.h>
#include <stdlib.h>

/**
 * power - Calculate the power of number.
 * @param base: Base value.
 * @param exponent: Exponent value.
 *
 * @return base raised to the power exponent.
 */
double power(double base, int exponent)
{
    int result = base;
    int i;
    
    if (exponent == 0) {
        return 1;
    }
    
    for(i = 1; i < exponent; ++i){
        result = result * base;
    }

    return result;
}

int main(int argc, char *argv[])
{
    if (argc < 3){
        printf("Usage: %s base exponent \n", argv[0]);
        return 1;
    }
    double base = atof(argv[1]);
    int exponent = atoi(argv[2]);
    double result = power(base, exponent);
    printf("%g ^ %d is %g\n", base, exponent, result);
    return 0;
}
```

### 编写 CMakeLists.txt

首先编写 CMakeLists.txt 文件，并保存在与 main.cc 源文件同个目录下：

```cmake
# CMake最低版本号要求
cmake_minimum_required(VERSION 2.8)

# 项目信息
project(Deom1)

# 指定生成目标
add_executable(Demo main.cc)
```

CMakeLists.txt 的语法比较简单，由命令、注释和空格组成，其中命令是不区分大小写的。符号 `#` 后面的内容被认为是注释。命令由命令名称、小括号和参数组成，参数之间使用空格进行间隔。

对于上面的 CMakeLists.txt 文件，依次出现了几个命令：

1. `cmake_minimum_required`：指定运行此配置文件所需的 CMake 的最低版本；
2. `project`：参数值是 `Demo1`，该命令表示项目的名称是 `Demo1` 。
3. `add_executable`： 将名为 main.cc 的源文件编译成一个名称为 Demo 的可执行文件。

### 编译项目

之后，在当前目录执行 `cmake .` ，得到 Makefile 后再使用 `make` 命令编译得到 Demo1 可执行文件。

```bash
[ehome@xman Demo1]$ cmake .
-- The C compiler identification is GNU 4.8.2
-- The CXX compiler identification is GNU 4.8.2
-- Check for working C compiler: /usr/sbin/cc
-- Check for working C compiler: /usr/sbin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working CXX compiler: /usr/sbin/c++
-- Check for working CXX compiler: /usr/sbin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Configuring done
-- Generating done
-- Build files have been written to: /home/ehome/Documents/programming/C/power/Demo1
[ehome@xman Demo1]$ make
Scanning dependencies of target Demo
[100%] Building C object CMakeFiles/Demo.dir/main.cc.o
Linking C executable Demo
[100%] Built target Demo
[ehome@xman Demo1]$ ./Demo 5 4
5 ^ 4 is 625
[ehome@xman Demo1]$ ./Demo 7 3
7 ^ 3 is 343
[ehome@xman Demo1]$ ./Demo 2 10
2 ^ 10 is 1024
```

## 多个源文件

### 同一目录，多个源文件

本小节对应的源代码所在目录：[Demo2](https://github.com/myhhub/cmake-project/tree/master/Demo2)。

上面的例子只有单个源文件。现在假如把 `power` 函数单独写进一个名为 `MathFunctions.c` 的源文件里，使得这个工程变成如下的形式：

```bash
./Demo2
    |
    +--- main.cc
    |
    +--- MathFunctions.cc
    |
    +--- MathFunctions.h

```

这个时候，CMakeLists.txt 可以改成如下的形式：

```cmake
# CMake最小版本号要求
cmake_minimum_required(VERSION 2.8)

# 项目信息
project(Demo2)

# 指定生成目标
add_executable(Demo main.cc MathFunctions.cc)
```

唯一的改动只是在 `add_executable` 命令中增加了一个 `MathFunctions.cc` 源文件。这样写当然没什么问题，但是如果源文件很多，把所有源文件的名字都加进去将是一件烦人的工作。更省事的方法是使用 `aux_source_directory` 命令，该命令会查找指定目录下的所有源文件，然后将结果存进指定变量名。其语法如下：

```cmake
aux_source_directory(<dir> <variable>)
```

因此可以修改CMakeLists.txt如下：

```cmake
# CMake 最低版本号要求
cmake_minimum_required (VERSION 2.8)

# 项目信息
project (Demo2)

# 查找当前目录下的所有源文件
# 并将名称保存到DIR_SRCS变量
aux_source_directory(. DIR_SRCS)
# 指定生成目标
add_executable(Demo ${DIR_SRCS})
```

这样，CMake 会将当前目录所有源文件的文件名赋值给变量 `DIR_SRCS` ，再指示变量 `DIR_SRCS` 中的源文件需要编译成一个名称为 Demo 的可执行文件。



### 多个目录，多个源文件

本小节对应的源代码所在目录：[Demo3](https://github.com/myhhub/cmake-project/tree/master/Demo3)。

现在进一步将 MathFunctions.h 和 MathFunctions.cc 文件移动到 math 目录下。

```bash
./Demo3
    |
    +--- main.cc
    |
    +--- math/
          |
          +--- MathFunctions.cc
          |
          +--- MathFunctions.h

```

对于这种情况，需要分别在项目根目录 Demo3 和 math 目录里各编写一个 CMakeLists.txt 文件。为了方便，我们可以先将 math 目录里的文件编译成静态库再由 main 函数调用。

根目录中的 CMakeLists.txt ：

```cmake
# CMake 最低版本号要求
cmake_minimum_required (VERSION 2.8)

# 项目信息
project (Demo3)

# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_SRCS 变量
aux_source_directory(. DIR_SRCS)

# 添加 math 子目录
add_subdirectory(math)

# 指定生成目标
add_executable(Demo main.cc)

# 添加链接库
target_link_libraries(Demo MathFunctions)
```

该文件添加了下面的内容: 第3行，使用命令 `add_subdirectory` 指明本项目包含一个子目录 math，这样 math 目录下的 CMakeLists.txt 文件和源代码也会被处理 。第6行，使用命令 `target_link_libraries` 指明可执行文件 main 需要连接一个名为 MathFunctions 的链接库 。

子目录中的 CMakeLists.txt：

```cmake
# 查找当前目录下的所有源文件
# 并将名称保存到DIR_LIB_SRCS变量
aux_source_directory(. DIR_LIB_SRCS)

# 生成链接库
add_library(MathFunctions ${DIR_LIB_SRCS})
```

在该文件中使用命令 `add_library` 将 src 目录中的源文件编译为静态链接库。



## 自定义编译选项

本节对应的源代码所在目录：[Demo4](https://github.com/myhhub/cmake-project/tree/master/Demo4)。

CMake 允许为项目增加编译选项，从而可以根据用户的环境和需求选择最合适的编译方案。

例如，可以将 MathFunctions 库设为一个可选的库，如果该选项为 `ON` ，就使用该库定义的数学函数来进行运算。否则就调用标准库中的数学函数库。

### 修改 CMakeLists 文件

我们要做的第一步是在顶层的 CMakeLists.txt 文件中添加该选项：

```cmake
# CMake 最低版本号要求
cmake_minimum_required (VERSION 2.8)

# 项目信息
project (Demo4)

# 加入一个配置头文件，用于处理 CMake 对源码的设置
configure_file(
"${PROJECT_SOURCE_DIR}/config.h.in"
"${PROJECT_BINARY_DIR}/config.h"
)
# 是否使用自己的MathFunction库
option(USE_MYMATH
"Use provided math implementation" ON)

# 是否加入MathFunction库
if(USE_MYMATH)
	include_directories("${PROJECT_SOURCE_DIR}/math")
	add_subdirectory(math)
	set(EXTRA_LIBS ${EXTRA_LIBS} MathFunctions)
endif(USE_MYMATH)

# 查找大哥前目录下的所有源文件
# 并将名称保存到DIR_SRCS变量
aux_source_directory(. DIR_SRCS)

# 指定生成目标
add_executable(Demo ${DIR_SRCS})
target_link_libraries(Demo ${EXTRA_LIBS})
```

其中：

1. 第7行的 `configure_file` 命令用于加入一个配置头文件 config.h ，这个文件由 CMake 从 config.h.in 生成，通过这样的机制，将可以通过预定义一些参数和变量来控制代码的生成。
2. 第13行的 `option` 命令添加了一个 `USE_MYMATH` 选项，并且默认值为 `ON` 。
3. 第17行根据 `USE_MYMATH` 变量的值来决定是否使用我们自己编写的 MathFunctions 库。



### 修改 main.cc 文件

之后修改 main.cc 文件，让其根据 `USE_MYMATH` 的预定义值来决定是否调用标准库还是 MathFunctions 库：

```c++
#include <stdio.h>
#include <stdlib.h>
#include "config.h"

#ifdef USE_MYMATH
  #include "math/MathFunctions.h"
#else
  #include <math.h>
#endif


int main(int argc, char *argv[])
{
    if (argc < 3){
        printf("Usage: %s base exponent \n", argv[0]);
        return 1;
    }
    double base = atof(argv[1]);
    int exponent = atoi(argv[2]);
    
#ifdef USE_MYMATH
    printf("Now we use our own Math library. \n");
    double result = power(base, exponent);
#else
    printf("Now we use the standard library. \n");
    double result = pow(base, exponent);
#endif
    printf("%g ^ %d is %g\n", base, exponent, result);
    return 0;
}
```

### 编写 config.h.in 文件

上面的程序值得注意的是第2行，这里引用了一个 config.h 文件，这个文件预定义了 `USE_MYMATH` 的值。但我们并不直接编写这个文件，为了方便从 CMakeLists.txt 中导入配置，我们编写一个 config.h.in 文件，内容如下：

```ini
#cmakedefine USE_MYMATH
```

这样 CMake 会自动根据 CMakeLists 配置文件中的设置自动生成 config.h 文件。

### 编译项目

现在编译一下这个项目，为了便于交互式的选择该变量的值，可以使用 `ccmake` 命令（也可以使用 `cmake -i` 命令，该命令会提供一个会话式的交互式配置界面）：

CMake的交互式配置界面

从中可以找到刚刚定义的 `USE_MYMATH` 选项，按键盘的方向键可以在不同的选项窗口间跳转，按下 `enter` 键可以修改该选项。修改完成后可以按下 `c` 选项完成配置，之后再按 `g` 键确认生成 Makefile 。ccmake 的其他操作可以参考窗口下方给出的指令提示。

我们可以试试分别将 `USE_MYMATH` 设为 `ON` 和 `OFF` 得到的结果：

#### USE_MYMATH 为 ON

运行结果：

```bash
[ehome@xman Demo4]$ ./Demo
Now we use our own MathFunctions library. 
 7 ^ 3 = 343.000000
 10 ^ 5 = 100000.000000
 2 ^ 10 = 1024.000000
```

此时 config.h 的内容为：

```c++
#define USE_MYMATH
```

#### USE_MYMATH 为 OFF

运行结果：

```bash
[ehome@xman Demo4]$ ./Demo
Now we use the standard library. 
 7 ^ 3 = 343.000000
 10 ^ 5 = 100000.000000
 2 ^ 10 = 1024.000000
```

此时 config.h 的内容为：

```c++
/* #undef USE_MYMATH */
```

## 安装和测试

本节对应的源代码所在目录：[Demo5](https://github.com/myhhub/cmake-project/tree/master/Demo5)。

CMake 也可以指定安装规则，以及添加测试。这两个功能分别可以通过在产生 Makefile 后使用 `make install` 和 `make test` 来执行。在以前的 GNU Makefile 里，你可能需要为此编写 `install` 和 `test` 两个伪目标和相应的规则，但在 CMake 里，这样的工作同样只需要简单的调用几条命令。

### 定制安装规则

首先先在 math/CMakeLists.txt 文件里添加下面两行：

```cmake
# 指定MathFunction库的安装路径
install(TARGETS MathFunctions DESTINATION bin)
install(FILES MathFunctions.h DESTINATION include)
```

指明 MathFunctions 库的安装路径。之后同样修改根目录的 CMakeLists 文件，在末尾添加下面几行：

```cmake
# 指定安装路径
install (TARGETS Demo DESTINATION bin)
install (FILES "${PROJECT_BINARY_DIR}/config.h"
         DESTINATION include)
```

通过上面的定制，生成的 Demo 文件和 MathFunctions 函数库 libMathFunctions.o 文件将会被复制到 `/usr/local/bin` 中，而 MathFunctions.h 和生成的 config.h 文件则会被复制到 `/usr/local/include` 中。我们可以验证一下（顺带一提的是，这里的 `/usr/local/` 是默认安装到的根目录，可以通过修改 `CMAKE_INSTALL_PREFIX` 变量的值来指定这些文件应该拷贝到哪个根目录）：

```bash
[ehome@xman Demo5]$ sudo make install
[ 50%] Built target MathFunctions
[100%] Built target Demo
Install the project...
-- Install configuration: ""
-- Installing: /usr/local/bin/Demo
-- Installing: /usr/local/include/config.h
-- Installing: /usr/local/bin/libMathFunctions.a
-- Up-to-date: /usr/local/include/MathFunctions.h
[ehome@xman Demo5]$ ls /usr/local/bin
Demo  libMathFunctions.a
[ehome@xman Demo5]$ ls /usr/local/include
config.h  MathFunctions.h
```

### 为工程添加测试

添加测试同样很简单。CMake 提供了一个称为 CTest 的测试工具。我们要做的只是在项目根目录的 CMakeLists 文件中调用一系列的 `add_test` 命令。

```cmake
# 启用测试
enable_testing()

# 测试程序是否成功运行
add_test (test_run Demo 5 2)

# 测试帮助信息是否可以正常提示
add_test (test_usage Demo)
set_tests_properties (test_usage
  PROPERTIES PASS_REGULAR_EXPRESSION "Usage: .* base exponent")

# 测试 5 的平方
add_test (test_5_2 Demo 5 2)

set_tests_properties (test_5_2
 PROPERTIES PASS_REGULAR_EXPRESSION "is 25")

# 测试 10 的 5 次方
add_test (test_10_5 Demo 10 5)

set_tests_properties (test_10_5
 PROPERTIES PASS_REGULAR_EXPRESSION "is 100000")

# 测试 2 的 10 次方
add_test (test_2_10 Demo 2 10)

set_tests_properties (test_2_10
 PROPERTIES PASS_REGULAR_EXPRESSION "is 1024")
```

上面的代码包含了四个测试。第一个测试 `test_run` 用来测试程序是否成功运行并返回 0 值。剩下的三个测试分别用来测试 5 的 平方、10 的 5 次方、2 的 10 次方是否都能得到正确的结果。其中 `PASS_REGULAR_EXPRESSION` 用来测试输出是否包含后面跟着的字符串。

让我们看看测试的结果：

```bash
[ehome@xman Demo5]$ make test
Running tests...
Test project /home/ehome/Documents/programming/C/power/Demo5
    Start 1: test_run
1/4 Test #1: test_run .........................   Passed    0.00 sec
    Start 2: test_5_2
2/4 Test #2: test_5_2 .........................   Passed    0.00 sec
    Start 3: test_10_5
3/4 Test #3: test_10_5 ........................   Passed    0.00 sec
    Start 4: test_2_10
4/4 Test #4: test_2_10 ........................   Passed    0.00 sec

100% tests passed, 0 tests failed out of 4

Total Test time (real) =   0.01 sec
```

如果要测试更多的输入数据，像上面那样一个个写测试用例未免太繁琐。这时可以通过编写宏来实现：

```cmake
# 定义一个宏，用来简化测试工作
macro (do_test arg1 arg2 result)
  add_test (test_${arg1}_${arg2} Demo ${arg1} ${arg2})
  set_tests_properties (test_${arg1}_${arg2}
    PROPERTIES PASS_REGULAR_EXPRESSION ${result})
endmacro (do_test)
 
# 使用该宏进行一系列的数据测试
do_test (5 2 "is 25")
do_test (10 5 "is 100000")
do_test (2 10 "is 1024")
```

关于 CTest 的更详细的用法可以通过 `man 1 ctest` 参考 CTest 的文档。

## 支持 gdb

让 CMake 支持 gdb 的设置也很容易，只需要指定 `Debug` 模式下开启 `-g` 选项：

```cmake
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAG_DEBUG "ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
```

之后可以直接对生成的程序使用gdb调试。

## 添加环境检查

本节对应的源代码所在目录：[Demo6](https://github.com/myhhub/cmake-project/tree/master/Demo6)。

有时候可能要对系统环境做点检查，例如要使用一个平台相关的特性的时候。在这个例子中，我们检查系统是否自带 pow 函数。如果带有 pow 函数，就使用它；否则使用我们定义的 power 函数。

### 添加 CheckFunctionExists 宏

首先在顶层 CMakeLists 文件中添加 CheckFunctionExists.cmake 宏，并调用 `check_function_exists` 命令测试链接器是否能够在链接阶段找到 `pow` 函数。

```cmake
# 检查系统是否支持pow函数
include(${CMAKE_ROOT}/Modules/CheckFunctionExists.cmake)
check_function_exists(pow HAVE_POW)
```

将上面这段代码放在 `configure_file` 命令前。

### 预定义相关宏变量

接下来修改 config.h.in 文件，预定义相关的宏变量。

```cmake
// does the platform provide pow function?
#cmakedefine HAVE_POW
```

### 在代码中使用宏和函数

最后一步是修改 main.cc ，在代码中使用宏和函数：

```c++
#ifdef HAVE_POW
	printf("Now we use the standard library. \n");
#else
	printf("Now we use our own Math library. \n");
	double result = power(base, exponent);
#endif
```

## 添加版本号

本节对应的源代码所在目录：[Demo7](https://github.com/myhhub/cmake-project/tree/master/Demo7)。

给项目添加和维护版本号是一个好习惯，这样有利于用户了解每个版本的维护情况，并及时了解当前所用的版本是否过时，或是否可能出现不兼容的情况。

首先修改顶层 CMakeLists 文件，在 `project` 命令之后加入如下两行：

```cmake
set(Demo_VERSION_MAJOR 1)
set(Demo_VERSION_MINOR 0)
```

分别指定当前的项目的主版本号和副版本号。

之后，为了在代码中获取版本信息，我们可以修改 config.h.in 文件，添加两个预定义变量：

```c++
// the configured options and settings for Tutorial
#define Demo_VERSION_MAJOR @Demo_VERSION_MAJOR@
#define Demo_VERSION_MINOR @Demo_VERSION_MINOR@
```

这样就可以直接在代码中打印版本信息了：

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "config.h"
#include "math/MathFunctions.h"

int main(int argc, char *argv[])
{
    if (argc < 3){
        // print version info
        printf("%s Version %d.%d\n",
            argv[0],
            Demo_VERSION_MAJOR,
            Demo_VERSION_MINOR);
        printf("Usage: %s base exponent \n", argv[0]);
        return 1;
    }
    double base = atof(argv[1]);
    int exponent = atoi(argv[2]);
    
#if defined (HAVE_POW)
    printf("Now we use the standard library. \n");
    double result = pow(base, exponent);
#else
    printf("Now we use our own Math library. \n");
    double result = power(base, exponent);
#endif
    
    printf("%g ^ %d is %g\n", base, exponent, result);
    return 0;
}
```

## 生成安装包

本节对应的源代码所在目录：[Demo8](https://github.com/myhhub/cmake-project/tree/master/Demo8)。

本节将学习如何配置生成各种平台上的安装包，包括二进制安装包和源码安装包。为了完成这个任务，我们需要用到 CPack ，它同样也是由 CMake 提供的一个工具，专门用于打包。

首先在顶层的 CMakeLists.txt 文件尾部添加下面几行：

```cmake
# 构建一个CPack安装包
include(InstallTequiredSystemLibraries)
set(CPACK_RESOURCE_FILE_LICENSE
"${CMAKE_CURRENT_SOURCE_DIR}/License.txt")
set(CPACK_PACKAGE_VERSION_MAJOR "${Demo_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${Demo_VERSION_MINOR}")
include(CPack)
```

上面的代码做了以下几个工作：

1. 导入InstallRequiredSystemLibraries模块，以便之后导入CPack模块；
2. 设置一些CPack相关变量，包括版权信息和版本信息，其中版本信息用了上一节定义的版本号；
3. 导入CPack模块。

接下来的工作是像往常一样构建工程，并执行`cpack` 命令。

- 生成二进制安装包：

```cmake
cpack -C CPackSourceConfig.cmake
```

我们可以试一下。在生成项目后，执行`cpack -C CPackConfig.cmake`命令：

```bash
[ehome@xman Demo8]$ cpack -C CPackSourceConfig.cmake
CPack: Create package using STGZ
CPack: Install projects
CPack: - Run preinstall target for: Demo8
CPack: - Install project: Demo8
CPack: Create package
CPack: - package: /home/ehome/Documents/programming/C/power/Demo8/Demo8-1.0.1-Linux.sh generated.
CPack: Create package using TGZ
CPack: Install projects
CPack: - Run preinstall target for: Demo8
CPack: - Install project: Demo8
CPack: Create package
CPack: - package: /home/ehome/Documents/programming/C/power/Demo8/Demo8-1.0.1-Linux.tar.gz generated.
CPack: Create package using TZ
CPack: Install projects
CPack: - Run preinstall target for: Demo8
CPack: - Install project: Demo8
CPack: Create package
CPack: - package: /home/ehome/Documents/programming/C/power/Demo8/Demo8-1.0.1-Linux.tar.Z generated.
```

此时会在该目录下创建 3 个不同格式的二进制包文件：

```bash
[ehome@xman Demo8]$ ls Demo8-*
Demo8-1.0.1-Linux.sh  Demo8-1.0.1-Linux.tar.gz  Demo8-1.0.1-Linux.tar.Z
```

这 3 个二进制包文件所包含的内容是完全相同的。我们可以执行其中一个。此时会出现一个由 CPack 自动生成的交互式安装界面：

```bash
[ehome@xman Demo8]$ sh Demo8-1.0.1-Linux.sh 
Demo8 Installer Version: 1.0.1, Copyright (c) Humanity
This is a self-extracting archive.
The archive will be extracted to: /home/ehome/Documents/programming/C/power/Demo8

If you want to stop extracting, please press <ctrl-C>.
The MIT License (MIT)

Copyright (c) 2013 Joseph Pan(http://hahack.com)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


Do you accept the license? [yN]: 
y
By default the Demo8 will be installed in:
  "/home/ehome/Documents/programming/C/power/Demo8/Demo8-1.0.1-Linux"
Do you want to include the subdirectory Demo8-1.0.1-Linux?
Saying no will install in: "/home/ehome/Documents/programming/C/power/Demo8" [Yn]: 
y

Using target directory: /home/ehome/Documents/programming/C/power/Demo8/Demo8-1.0.1-Linux
Extracting, please wait...

Unpacking finished successfully
```

完成后提示安装到了 Demo8-1.0.1-Linux 子目录中，我们可以进去执行该程序：

```bash
[ehome@xman Demo8]$ ./Demo8-1.0.1-Linux/bin/Demo 5 2
Now we use our own Math library. 
5 ^ 2 is 25
```

关于 CPack 的更详细的用法可以通过 `man 1 cpack` 参考 CPack 的文档。

## 将其他平台的项目迁移到 CMake

CMake 可以很轻松地构建出在适合各个平台执行的工程环境。而如果当前的工程环境不是 CMake ，而是基于某个特定的平台，是否可以迁移到 CMake 呢？答案是可能的。下面针对几个常用的平台，列出了它们对应的迁移方案。

### autotools

- [am2cmake](https://projects.kde.org/projects/kde/kdesdk/kde-dev-scripts/repository/revisions/master/changes/cmake-utils/scripts/am2cmake) 可以将 autotools 系的项目转换到 CMake，这个工具的一个成功案例是 KDE 。
- [Alternative Automake2CMake](http://emanuelgreisen.dk/stuff/kdevelop_am2cmake.php.tgz) 可以转换使用 automake 的 KDevelop 工程项目。
- [Converting autoconf tests](http://www.cmake.org/Wiki/GccXmlAutoConfHints)

### qmake

- [qmake converter](http://www.cmake.org/Wiki/CMake:ConvertFromQmake) 可以转换使用 QT 的 qmake 的工程。

### Visual Studio

- [vcproj2cmake.rb](http://vcproj2cmake.sf.net/) 可以根据 Visual Studio 的工程文件（后缀名是 `.vcproj` 或 `.vcxproj`）生成 CMakeLists.txt 文件。
- [vcproj2cmake.ps1](http://nberserk.blogspot.com/2010/11/converting-vc-projectsvcproj-to.html) vcproj2cmake 的 PowerShell 版本。
- [folders4cmake](http://sourceforge.net/projects/folders4cmake/) 根据 Visual Studio 项目文件生成相应的 “source_group” 信息，这些信息可以很方便的在 CMake 脚本中使用。支持 Visual Studio 9/10 工程文件。

### CMakeLists.txt 自动推导

- [gencmake](http://websvn.kde.org/trunk/KDE/kdesdk/cmake/scripts/) 根据现有文件推导 CMakeLists.txt 文件。
- [CMakeListGenerator](http://www.vanvelzensoftware.com/postnuke/index.php?name=Downloads&req=viewdownload&cid=7) 应用一套文件和目录分析创建出完整的 CMakeLists.txt 文件。仅支持 Win32 平台。

## 相关链接

1. [官方主页](http://www.cmake.org/)
2. [官方文档](http://www.cmake.org/cmake/help/cmake2.4docs.html)
3. [官方教程](http://www.cmake.org/cmake/help/cmake_tutorial.html)
4. [Wiki](http://www.cmake.org/Wiki/CMake#Basic_CMakeLists.txt_from-scratch-generator)
5. [FAQ](http://www.cmake.org/Wiki/CMake_FAQ)
6. [bug tracker](http://www.cmake.org/Bug)
7. 邮件列表：
   - [cmake on Gmane](http://dir.gmane.org/gmane.comp.programming.tools.cmake.user)
   - http://www.mail-archive.com/cmake@cmake.org/
   - [http://marc.info/?l=cmake](http://www.mail-archive.com/cmake@cmake.org/)
8. 其他推荐文章
   - [在 linux 下使用 CMake 构建应用程序](http://www.ibm.com/developerworks/cn/linux/l-cn-cmake/)
   - [cmake的一些小经验](http://www.cppblog.com/skyscribe/archive/2009/12/14/103208.aspx)
   - [Packaging Software with CPack](http://www.kitware.com/media/archive/kitware_quarterly0107.pdf)
   - [视频教程: 《Getting Started with CMake》](http://www.youtube.com/watch?v=CLvZTyji_Uw)



## 类似工具

- [SCons](http://scons.org/)：Eric S. Raymond、Timothee Besset、Zed A. Shaw 等大神力荐的项目架构工具。和 CMake 的最大区别是使用 Python 作为执行脚本。

------

1. [这个页面](http://www.cmake.org/Wiki/CMake_Projects)详细罗列了使用 CMake 的知名项目。







# CMake 完整使用教程 之一 配置环境

[CMake 完整使用教程 之一 配置环境 | 来唧唧歪歪(Ljjyy.com) - 多读书多实践，勤思考善领悟](https://www.ljjyy.com/archives/2021/03/100651)

学习CMake之前，需要对系统进行设置，这样才能运行所有示例。

本章的主要内容有：

- 如何获取代码
- 如何在GNU/Linux、macOS和Windows上安装运行示例所需的所有工具
- 自动化测试如何工作
- 如何报告问题，并提出改进建议

我们会尽可能让初学者看懂本书的内容。不过，这本书并非完全适合零基础人士。我们假设，您对构建目标平台上可用的软件，及本地工具有基本的了解。有Git版本控制的经验，可与源码库进行“互动”(不是必需)。

## 获取代码

本书的源代码可以在GitHub上找到，网址是 https://github.com/dev-cafe/cmake-cookbook 。开源代码遵循MIT许可：只要原始版权和许可声明包含在软件/源代码的任何副本中，可以以任何方式重用和重新混合代码。许可的全文可以在 https://opensource.org/licenses/MIT 中看到。

为了测试源码，需要使用Git获取代码：

- 主要的GNU/Linux发行版都可以通过包管理器安装Git。也可以从Git项目网站 [https://git-scm.com](https://git-scm.com/) 下载二进制发行版，进行安装。
- MacOS上，可以使用自制或MacPorts安装Git。
- Windows上，可以从Git项目网站( [https://git-scm.com](https://git-scm.com/) )下载Git可执行安装文件。

可以通过GitHub桌面客户端访问这些示例，网址为 [https://desktop.github.com](https://desktop.github.com/) 。

另一种选择是从 https://github.com/dev-cafe/cmake-cookbook 下载zip文件。

安装Git后，可以将远程库克隆到本地计算机，如下所示：

```bash
$ git clone https://github.com/dev-cafe/cmake-cookbook.git
```

这将创建一个名为`cmake-cookbook`的文件夹。本书内容与源码的章节对应，书中章节的编号和源码的顺序相同。

在GNU/Linux、MacOS和Windows上，使用最新的持续集成进行测试。我们会在之后讨论测试的设置。

我们用标签v1.0标记了与本书中打印的示例相对应的版本。为了与书中内容对应，可以如下获取此特定版本：

```bash
$ git clone --single-branch -b v1.0 https://github.com/dev-cafe/cmake-cookbook.git
```

我们希望收到Bug修复，并且GitHub库将继续发展。要获取更新，可以选择库的master分支。

## Docker镜像

在Docker中进行环境搭建，无疑是非常方便的(依赖项都已经安装好了)。我们的Docker镜像是基于Ubuntu 18.04的镜像制作，您可以按照官方文档[https://docs.docker.com](https://docs.docker.com/) 在您的操作系统上安装Docker。

Docker安装好后，您可以下载并运行我们的镜像，然后可以对本书示例进行测试:

```bash
$ docker run -it devcafe/cmake-cookbook_ubuntu-18.04
$ git clone https://github.com/dev-cafe/cmake-cookbook.git
$ cd cmake-cookbook
$ pipenv install --three
$ pipenv run python testing/collect_tests.py 'chapter-*/recipe-*'
```

## 安装必要的软件

与在Docker中使用不同，另一种选择是直接在主机操作系统上安装依赖项。为此，我们概括了一个工具栈，可以作为示例的基础。您必须安装以下组件：

1. CMake
2. 编译器
3. 自动化构建工具
4. Python

我们还会详细介绍，如何安装所需的某些依赖项。

### 获取CMake

本书要使用的CMake最低需要为3.5。只有少数示例，演示了3.5版之后引入的新功能。每个示例都有提示，指出示例代码在哪里可用，以及所需的CMake的最低版本。提示信息如下:

**NOTE**:*这个示例的代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-03/recipe10 中找到，其中包括一个C示例。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行了测试。*

有些(如果不是大多数)示例仍然适用于较低版本的CMake。但是，我们没有测试过这个。我们认为CMake 3.5是大多数系统和发行版的默认软件，而且升级CMake也没什么难度。

CMake可以以多种方式安装。下载并提取由Kitware维护的二进制发行版，可以在所有平台上运行，下载页面位于 https://cmake.org/download/ 。

大多数GNU/Linux发行版都在包管理器中提供了CMake。然而，在一些发行版中，版本可能比较旧，因此下载由Kitware提供的二进制文件当然是首选。下面的命令将从CMake打包的版本中下载并安装在`$HOME/Deps/CMake`(根据您的偏好调整此路径)下的CMake 3.5.2：

```bash
$ cmake_version="3.5.2"
$ target_path=$HOME/Deps/cmake/${cmake_version}
$ cmake_url="https://cmake.org/files/v${cmake_version%.*}/cmake-${cmake_version}-Linux-x86_64.tar.gz"
$ mkdir -p "${target_path}"
$ curl -Ls "${cmake_url}" | tar -xz -C "${target_path}" --strip-components=1
$ export PATH=$HOME/Deps/cmake/${cmake_version}/bin${PATH:+:$PATH}
$ cmake --version
```

macOS获取最新版本的CMake：

```bash
$ brew upgrade cmake
```

Windows上，可以使用Visual Studio 2017，它提供了CMake支持。Visual Studio 2017的安装记录在第13章，*可选生成器和交叉编译*，示例技巧1，*使用Visual Studio 2017构建CMake项目*。

或者，可以从 [https://www.msys2.org](https://www.msys2.org/) 下载MSYS2安装程序，按照其中给出的说明更新包列表，然后使用包管理器`pacman`安装CMake。下面的代码正在构建64位版本：

```bash
$ pacman -S mingw64/mingw-w64-x86_64-cmake
```

对于32位版本，请使用以下代码(为了简单起见，我们以后只会提到64位版本)：

```bash
$ pacman -S mingw64/mingw-w64-i686-cmake
```

MSYS2的另一个特性是在Windows上提供了一个终端，比较像Unix操作系统上的终端，提供可用的开发环境。

### 编译器

我们将需要C++、C和Fortran的编译器。编译器的版本需要比较新，因为我们需要在大多数示例中支持最新的语言标准。CMake为来自商业和非商业供应商的许多编译器，提供了非常好的支持。为了让示例始终能够跨平台，并尽可能独立于操作系统，我们使用了开源编译器:

- GNU/Linux上，GNU编译器集合(GCC)是直接的选择。它是免费的，适用于所有发行版。例如，在Ubuntu上，可以安装以下编译器：

```bash
$ sudo apt-get install g++ gcc gfortran
```

- 在LLVM家族中，Clang也是C++和C编译器的一个很好的选择：

```bash
$ sudo apt-get install clang clang++ gfortran
```

- macOS上，XCode附带的LLVM编译器适用于C++和C。我们在macOS测试中使用了GCC的Fortran编译器。GCC编译器必须使用包管理器单独安装：

```bash
$ brew install gcc
```

Windows上，可以使用Visual Studio测试C++和C示例。或者，可以使用MSYS2安装程序，MSYS2环境中(对于64位版本)使用以下单个命令安装整个工具链，包括C++、C和Fortran编译器：

```bash
$ pacman -S mingw64/mingw-w64-x86_64-toolchain
```

### 自动化构建工具

自动化构建工具为示例中的项目提供构建和链接的基础设施，最终会安装和使用什么，很大程度上取决于操作系统：

- GNU/Linux上，GNU Make(很可能)在安装编译器时自动安装。
- macOS上，XCode将提供GNU Make。
- Windows上，Visual Studio提供了完整的基础设施。MSYS2环境中，GNU Make作为mingw64/mingw-w64-x86_64工具链包的一部分，进行安装。

为了获得最大的可移植性，我们尽可能使示例不受这些系统相关细节的影响。这种方法的优点是配置、构建和链接，是每个编译器的*固有特性*。

Ninja是一个不错的自动化构建工具，适用于GNU/Linux、macOS和Windows。Ninja注重速度，特别是增量重构。为GNU/Linux、macOS和Windows预先打包的二进制文件可以在GitHub库中找到，网址是 https://github.com/ninja-build/ninja/releases 。

Fortran项目中使用CMake和Ninja需要注意。使用CMake 3.7.2或更高版本是必要的，Kitware还有维护Ninja，相关包可以在 https://github.com/Kitware/ninja/releases 上找到。

在GNU/Linux上，可以使用以下一系列命令安装Ninja：

```bash
$ mkdir -p ninja
$ ninja_url="https://github.com/Kitware/ninja/releases/download/v1.8.2.g3bbbe.kitware.dyndep-1.jobserver-1/ninja-1.8.2.g3bbbe.kitware.dyndep-1.jobserver-1_x86_64-linux-gnu.tar.gz"
$ curl -Ls ${ninja_url} | tar -xz -C ninja --strip-components=1
$ export PATH=$HOME/Deps/ninja${PATH:+:$PATH}
```

Windows上，使用MSYS2环境(假设是64位版本)执行以下命令：

```bash
$ pacman -S mingw64/mingw-w64-x86_64-ninja
```

**NOTE**:*我们建议阅读这篇文章 http://www.aosabook.org/en/posa/ninja.html ，里面是对NInja编译器的历史和设计的选择，进行启发性的讨论。*

### Python

本书主要关于CMake，但是其中的一些方法，需要使用Python。因此，也需要对Python进行安装：解释器、头文件和库。Python 2.7的生命周期结束于2020年，因此我们将使用Python 3.5。

在Ubuntu 14.04 LTS上(这是Travis CI使用的环境，我们后面会讨论)，Python 3.5可以安装如下：

```bash
sudo apt-get install python3.5-dev
```

Windows可使用MSYS2环境，Python安装方法如下(假设是64位版本):

```bash
$ pacman -S mingw64/mingw-w64-x86_64-python3
$ pacman -S mingw64/mingw-w64-x86_64-python3-pip
$ python3 -m pip install pipenv
```

为了运行已经写好的测试机制，还需要一些特定的Python模块。可以使用包管理器在系统范围内安装这些包，也可以在隔离的环境中安装。建议采用后一种方法：

- 可以在不影响系统环境的情况下，将安装包进行清理/安装。
- 可以在没有管理员权限的情况下安装包。
- 可以降低软件版本和依赖项冲突的风险。
- 为了复现性，可以更好地控制包的依赖性。

为此，我们准备了一个`Pipfile`。结合`pipfile.lock`，可以使用`Pipenv`( [http://pipenv.readthedocs](http://pipenv.readthedocs/) )。创建一个独立的环境，并安装所有包。要为示例库创建此环境，可在库的顶层目录中运行以下命令：

```bash
$ pip install --user pip pipenv --upgrade
$ pipenv install --python python3.5
```

执行`pipenv shell`命令会进入一个命令行环境，其中包含特定版本的Python和可用的包。执行`exit`将退出当前环境。当然，还可以使用`pipenv run`在隔离的环境中直接执行命令。

或者，可以将库中的`requirements.txt`文件与`Virtualenv`( http://docs.pythonguide.org/en/latest/dev/virtualenvs/ )和`pip`结合使用，以达到相同的效果：

```bash
$ virtualenv --python=python3.5 venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

可以使用`deactivate`命令退出虚拟环境。

另一种选择是使用`Conda`环境，我们建议安装`Miniconda`。将把最新的`Miniconda`安装到GNU/Linux的`$HOME/Deps/conda`目录(从 https://repo.continuum.io/miniconda/miniconda3-latestlinux-x86_64.sh 下载)或macOS(从 https://repo.continuum.io/miniconda/miniconda3-latestmacosx-x86_64.sh 下载)：

```bash
$ curl -Ls https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh > miniconda.sh
$ bash miniconda.sh -b -p "$HOME"/Deps/conda &> /dev/null
$ touch "$HOME"/Deps/conda/conda-meta/pinned
$ export PATH=$HOME/Deps/conda/bin${PATH:+:$PATH}
$ conda config --set show_channel_urls True
$ conda config --set changeps1 no
$ conda update --all
$ conda clean -tipy
```

Windows上，可以从 https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe 下载最新的`Miniconda`。该软件包可以使用`PowerShell`安装，如下:

```bash
$basedir = $pwd.Path + "\"
$filepath = $basedir + "Miniconda3-latest-Windows-x86_64.exe"
$Anaconda_loc = "C:\Deps\conda"
$args = "/InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /S /D=$Anaconda_loc"
Start-Process -FilePath $filepath -ArgumentList $args -Wait -Passthru
$conda_path = $Anaconda_loc + "\Scripts\conda.exe"
$args = "config --set show_channel_urls True"
Start-Process -FilePath "$conda_path" -ArgumentList $args -Wait -Passthru
$args = "config --set changeps1 no"
Start-Process -FilePath "$conda_path" -ArgumentList $args -Wait -Passthru
$args = "update --all"
Start-Process -FilePath "$conda_path" -ArgumentList $args -Wait -Passthru
$args = "clean -tipy"
Start-Process -FilePath "$conda_path" -ArgumentList $args -Wait -Passthru
```

安装了`Conda`后, Python模块可以按如下方式安装:

```bash
$ conda create -n cmake-cookbook python=3.5
$ conda activate cmake-cookbook
$ conda install --file requirements.txt
```

执行`conda deactivate`将退出`conda`的环境。

### 依赖软件

有些示例需要额外的依赖，这些软件将在这里介绍。

####  BLAS和LAPACK

大多数Linux发行版都为BLAS和LAPACK提供包。例如，在Ubuntu 14.04 LTS上，您可以运行以下命令：

```bash
$ sudo apt-get install libatlas-dev liblapack-dev liblapacke-dev
```

macOS上，XCode附带的加速库可以满足我们的需要。

Windows使用MSYS2环境，可以按如下方式安装这些库(假设是64位版本)：

```bash
$ pacman -S mingw64/mingw-w64-x86_64-openblas
```

或者，可以从GitHub ( https://github.com/referlapack/lapack )下载BLAS和LAPACK的参考实现，并从源代码编译库。商业供应商为平台提供安装程序，安装包中有BLAS和LAPACK相关的API。

#### 消息传递接口(MPI)

MPI有许多商业和非商业实现。这里，安装免费的非商业实现就足够了。在Ubuntu 14.04 LTS上，我们推荐`OpenMPI`。可使用以下命令安装：

```bash
$ sudo apt-get install openmpi-bin libopenmpi-dev
```

在macOS上，`Homebrew`发布了`MPICH`：

```bash
$ brew install mpich
```

还可以从 https://www.open-mpi.org/software/ 上获取源代码，编译`OpenMPI`。 对于Windows，Microsoft MPI可以通过 https://msdn.microsoft.com/en-us/library/bb524831(v=vs.85).aspx 下载安装。

#### 线性代数模板库

一些示例需要线性代数模板库，版本为3.3或更高。如果包管理器不提供`Eigen`，可以使用在线打包源([http://eigen.tuxfamily.org](http://eigen.tuxfamily.org/) )安装它。例如，在GNU/Linux和macOS上，可以将`Eigen`安装到`$HOME/Deps/Eigen`目录:

```bash
$ eigen_version="3.3.4"
$ mkdir -p eigen
$ curl -Ls http://bitbucket.org/eigen/eigen/get/${eigen_version}.tar.gz | tar -xz -C eigen --strip-components=1
$ cd eigen
$ cmake -H. -Bbuild_eigen -
DCMAKE_INSTALL_PREFIX="$HOME/Deps/eigen" &> /dev/null
$ cmake --build build_eigen -- install &> /dev/null
```

#### Boost库

`Boost`库适用于各种操作系统，大多数Linux发行版都通过它们的包管理器提供该库的安装。例如，在Ubuntu 14.04 LTS上，`Boost`文件系统库、`Boost Python`库和`Boost`测试库可以通过以下命令安装：

```bash
$ sudo apt-get install libboost-filesystem-dev libboost-python-dev libboost-test-dev
```

对于macOS, `MacPorts`和自制程序都为最新版本的`Boost`提供了安装包。我们在macOS上的测试设置安装`Boost`如下：

```bash
$ brew cask uninstall --force oclint
$ brew uninstall --force --ignore-dependencies boost
$ brew install boost
$ brew install boost-python3
```

Windows的二进制发行版也可以从`Boost`网站 [http://www.boost.org](http://www.boost.org/) 下载。或者，可以从 [https://www.boost.org](https://www.boost.org/) 下载源代码，并自己编译`Boost`库。

#### 交叉编译器

在类Debian/Ubuntu系统上，可以使用以下命令安装交叉编译器：

```bash
$ sudo apt-get install gcc-mingw-w64 g++-mingw-w64 gfortran-mingw-w64
```

在macOS上，使用`Brew`，可以安装以下交叉编译器：

```bash
$ brew install mingw-w64
```

其他包管理器提供相应的包。使用打包的跨编译器的另一种方法，是使用M交叉环境( [https://mxe.cc](https://mxe.cc/) )，并从源代码对其进行构建。

#### ZeroMQ, pkg-config, UUID和Doxygen

Ubuntu 14.04 LTS上，这些包可以安装如下：

```bash
$ sudo apt-get install pkg-config libzmq3-dev doxygen graphviz-dev uuid-dev
```

macOS上，我们建议使用`Brew`安装：

```bash
$ brew install ossp-uuid pkg-config zeromq doxygen
```

`pkg-config`程序和`UUID`库只在类Unix系统上可用。 Windows上使用MSYS2环境，可以按如下方式安装这些依赖项(假设是64位版本)：

```bash
$ pacman -S mingw64/mingw-w64-x86_64-zeromq
$ pacman -S mingw64/mingw-w64-x86_64-pkg-config
$ pacman -S mingw64/mingw-w64-x86_64-doxygen
$ pacman -S mingw64/mingw-w64-x86_64-graphviz
```

#### Conda的构建和部署

想要使用`Conda`打包的示例的话，需要`Miniconda`和`Conda`构建和部署工具。`Miniconda`的安装说明之前已经给出。要在GNU/Linux和macOS上安装`Conda`构建和部署工具，请运行以下命令:

```bash
$ conda install --yes --quiet conda-build anaconda-client jinja2 setuptools
$ conda clean -tipsy
$ conda info -a
```

这些工具也可以安装在Windows上:

```bash
$conda_path = "C:\Deps\conda\Scripts\conda.exe"
$args = "install --yes --quiet conda-build anaconda-client jinja2 setuptools"
Start-Process -FilePath "$conda_path" -ArgumentList $args -Wait -Passthru
$args = "clean -tipsy"
Start-Process -FilePath "$conda_path" -ArgumentList $args -Wait -Passthru
$args = "info -a"
Start-Process -FilePath "$conda_path" -ArgumentList $args -Wait -Passthru
```

## 测试环境

示例在下列持续集成(CI)上进行过测试：

- Travis( [https://travis-ci.org](https://travis-ci.org/) )用于GNU/Linux和macOS
- Appveyor( [https://www.appveyor.com](https://www.appveyor.com/) )用于Windows
- CircleCI ( [https://circleci.com](https://circleci.com/) )用于附加的GNU/Linux测试和商业编译器

CI服务的配置文件可以在示例库中找到( https://github.com/dev-cafe/cmake-cookbook/ ):

- Travis的配置文件为`travis.yml`
- Appveyor的配置文件为`.appveyor.yml`
- CircleCI的配置文件为`.circleci/config.yml`
- Travis和Appveyor的其他安装脚本，可以在`testing/dependencies`文件夹中找到。

**NOTE**:*GNU/Linux系统上，Travis使用CMake 3.5.2和CMake 3.12.1对实例进行测试。macOS系统上用CMake 3.12.1进行测试。Appveyor使用CMake 3.11.3进行测试。Circle使用CMake 3.12.1进行测试。*

测试机制是一组Python脚本，包含在`testing`文件夹中。脚本`collect_tests.py`将运行测试并报告它们的状态。示例也可以单独测试，也可以批量测试；`collect_tests.py`接受正则表达式作为命令行输入，例如:

```bash
$ pipenv run python testing/collect_tests.py 'chapter-0[1,7]/recipe-0[1,2,5]'
```

该命令将对第1章和第7章的示例1、2和5进行测试。

要获得更详细的输出，可以设置环境变量`VERBOSE_OUTPUT=ON`：

```bash
$ env VERBOSE_OUTPUT=ON pipenv run python testing/collect_tests.py 'chapter-*/recipe-*'
```



# CMake 完整使用教程 之二 从可执行文件到库



本章的主要内容有：

- 将单个源码文件编译为可执行文件
- 切换生成器
- 构建和连接静态库与动态库
- 用条件语句控制编译
- 向用户显示选项
- 指定编译器
- 切换构建类型
- 设置编译器选项
- 为语言设定标准
- 使用控制流进行构造

本章的示例将指导您完成构建代码所需的基本任务：编译可执行文件、编译库、根据用户输入执行构建操作等等。CMake是一个构建系统生成器，特别适合于独立平台和编译器。除非另有说明，否则所有配置都独立于操作系统，它们可以在GNU/Linux、macOS和Windows的系统下运行。

本书的示例主要为C++项目设计，并使用C++示例进行了演示，但CMake也可以用于其他语言的项目，包括C和Fortran。我们会尝试一些有意思的配置，其中包含了一些C++、C和Fortran语言示例。您可以根据自己喜好，选择性了解。有些示例是定制的，以突出在选择特定语言时需要面临的挑战。

## 将单个源文件编译为可执行文件

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-01 中找到，包含C++、C和Fortran示例。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

本节示例中，我们将演示如何运行CMake配置和构建一个简单的项目。该项目由单个源文件组成，用于生成可执行文件。我们将用C++讨论这个项目，您在GitHub示例库中可以找到C和Fortran的例子。

### 准备工作

我们希望将以下源代码编译为单个可执行文件：

```c++
#include <cstdlib>
#include <iostream>
#include <string>

std::string say_hello() { return std::string("Hello, CMake world!"); }

int main() {
  std::cout << say_hello() << std::endl;
  return EXIT_SUCCESS;
}
```

### 具体实施

除了源文件之外，我们还需要向CMake提供项目配置描述。该描述使用CMake完成，完整的文档可以在 https://cmake.org/cmake/help/latest/ 找到。我们把CMake指令放入一个名为`CMakeLists.txt`的文件中。

**NOTE**:*文件的名称区分大小写，必须命名为`CMakeLists.txt`，CMake才能够解析。*

具体步骤如下：

1. 用编辑器打开一个文本文件，将这个文件命名为`CMakeLists.txt`。
2. 第一行，设置CMake所需的最低版本。如果使用的CMake版本低于该版本，则会发出致命错误：

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
```

3. 第二行，声明了项目的名称(`recipe-01`)和支持的编程语言(CXX代表C++)：

```cmake
project(recipe-01 LANGUAGES CXX)
```

4. 指示CMake创建一个新目标：可执行文件`hello-world`。这个可执行文件是通过编译和链接源文件`hello-world.cpp`生成的。CMake将为编译器使用默认设置，并自动选择生成工具：

```cmake
add_executable(hello-world hello-world.cpp)
```

5. 将该文件与源文件`hello-world.cpp`放在相同的目录中。记住，它只能被命名为`CMakeLists.txt`。
6. 现在，可以通过创建`build`目录，在`build`目录下来配置项目：

```bash
$ mkdir -p build
$ cd build
$ cmake ..

-- The CXX compiler identification is GNU 8.1.0
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Configuring done
-- Generating done
-- Build files have been written to: /home/user/cmake-cookbook/chapter-01/recipe-01/cxx-example/build

```

7. 如果一切顺利，项目的配置已经在`build`目录中生成。我们现在可以编译可执行文件：

```bash
$ cmake --build .

Scanning dependencies of target hello-world
[ 50%] Building CXX object CMakeFiles/hello-world.dir/hello-world.cpp.o
[100%] Linking CXX executable hello-world
[100%] Built target hello-world
```

### 工作原理

示例中，我们使用了一个简单的`CMakeLists.txt`来构建“Hello world”可执行文件：

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(recipe-01 LANGUAGES CXX)
add_executable(hello-world hello-world.cpp)
```

**NOTE**:*CMake语言不区分大小写，但是参数区分大小写。*

**TIPS**:*CMake中，C++是默认的编程语言。不过，我们还是建议使用`LANGUAGES`选项在`project`命令中显式地声明项目的语言。*

要配置项目并生成构建器，我们必须通过命令行界面(CLI)运行CMake。CMake CLI提供了许多选项，`cmake -help`将输出以显示列出所有可用选项的完整帮助信息，我们将在书中对这些选项进行更多地了解。正如您将从`cmake -help`的输出中显示的内容，它们中的大多数选项会让你您访问CMake手册，查看详细信息。通过下列命令生成构建器：

```bash
$ mkdir -p build
$ cd build
$ cmake ..
```

这里，我们创建了一个目录`build`(生成构建器的位置)，进入`build`目录，并通过指定`CMakeLists.txt`的位置(本例中位于父目录中)来调用CMake。可以使用以下命令行来实现相同的效果：

```bash
$ cmake -H. -Bbuild
```

命令是跨平台的，使用了`-H`和`-B`为CLI选项。`-H`表示当前目录中搜索根`CMakeLists.txt`文件。`-Bbuild`告诉CMake在一个名为`build`的目录中生成所有的文件。

**NOTE**:*`cmake -H. -Bbuild`也属于CMake标准使用方式: https://cmake.org/pipermail/cmake-developers/2018-January/030520.html 。不过，我们将在本书中使用传统方法(创建一个构建目录，进入其中，并通过将CMake指向`CMakeLists.txt`的位置来配置项目)。*

运行`cmake`命令会输出一系列状态消息，显示配置信息：

```bash
$ cmake ..

-- The CXX compiler identification is GNU 8.1.0
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Configuring done
-- Generating done
-- Build files have been written to: /home/user/cmake-cookbook/chapter-01/recipe-01/cxx-example/build
```

**NOTE**:*在与`CMakeLists.txt`相同的目录中执行`cmake .`，原则上足以配置一个项目。然而，CMake会将所有生成的文件写到项目的根目录中。这将是一个源代码内构建，通常是不推荐的，因为这会混合源代码和项目的目录树。我们首选的是源外构建。*

CMake是一个构建系统生成器。将描述构建系统(如：Unix Makefile、Ninja、Visual Studio等)应当如何操作才能编译代码。然后，CMake为所选的构建系统生成相应的指令。默认情况下，在GNU/Linux和macOS系统上，CMake使用Unix Makefile生成器。Windows上，Visual Studio是默认的生成器。在下一个示例中，我们将进一步研究生成器，并在第13章中重新讨论生成器。

GNU/Linux上，CMake默认生成Unix Makefile来构建项目：

- `Makefile`: `make`将运行指令来构建项目。
- `CMakefile`：包含临时文件的目录，CMake用于检测操作系统、编译器等。此外，根据所选的生成器，它还包含特定的文件。
- `cmake_install.cmake`：处理安装规则的CMake脚本，在项目安装时使用。
- `CMakeCache.txt`：如文件名所示，CMake缓存。CMake在重新运行配置时使用这个文件。

要构建示例项目，我们运行以下命令：

```bash
$ cmake --build .
```

最后，CMake不强制指定构建目录执行名称或位置，我们完全可以把它放在项目路径之外。这样做同样有效：

```bash
$ mkdir -p /tmp/someplace
$ cd /tmp/someplace
$ cmake /path/to/source
$ cmake --build .
```

### 更多信息

官方文档 https://cmake.org/runningcmake/ 给出了运行CMake的简要概述。由CMake生成的构建系统，即上面给出的示例中的Makefile，将包含为给定项目构建目标文件、可执行文件和库的目标及规则。`hello-world`可执行文件是在当前示例中的唯一目标，运行以下命令：

```bash
$ cmake --build . --target help

The following are some of the valid targets for this Makefile:
... all (the default if no target is provided)
... clean
... depend
... rebuild_cache
... hello-world
... edit_cache
... hello-world.o
... hello-world.i
... hello-world.s
```

CMake生成的目标比构建可执行文件的目标要多。可以使用`cmake --build . --target <target-name>`语法，实现如下功能：

- **all**(或Visual Studio generator中的ALL_BUILD)是默认目标，将在项目中构建所有目标。
- **clean**，删除所有生成的文件。
- **rebuild_cache**，将调用CMake为源文件生成依赖(如果有的话)。
- **edit_cache**，这个目标允许直接编辑缓存。

对于更复杂的项目，通过测试阶段和安装规则，CMake将生成额外的目标：

- **test**(或Visual Studio generator中的**RUN_TESTS**)将在CTest的帮助下运行测试套件。我们将在第4章中详细讨论测试和CTest。
- **install**，将执行项目安装规则。我们将在第10章中讨论安装规则。
- **package**，此目标将调用CPack为项目生成可分发的包。打包和CPack将在第11章中讨论。

## 切换生成器

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-02 中找到，其中有一个C++、C和Fortran示例。该配置在CMake 3.5版(或更高版本)下测试没问题，并且已经在GNU/Linux、macOS和Windows上进行了测试。*

CMake是一个构建系统生成器，可以使用单个CMakeLists.txt为不同平台上的不同工具集配置项目。您可以在CMakeLists.txt中描述构建系统必须运行的操作，以配置并编译代码。基于这些指令，CMake将为所选的构建系统(Unix Makefile、Ninja、Visual Studio等等)生成相应的指令。我们将在第13章中重新讨论生成器。

### 准备工作

CMake针对不同平台支持本地构建工具列表。同时支持命令行工具(如Unix Makefile和Ninja)和集成开发环境(IDE)工具。用以下命令，可在平台上找到生成器名单，以及已安装的CMake版本：

```bash
$ cmake --help
```

这个命令的输出，将列出CMake命令行界面上所有的选项，您会找到可用生成器的列表。例如，安装了CMake 3.11.2的GNU/Linux机器上的输出：

```bash
Generators
The following generators are available on this platform:
Unix Makefiles = Generates standard UNIX makefiles.
Ninja = Generates build.ninja files.
Watcom WMake = Generates Watcom WMake makefiles.
CodeBlocks - Ninja = Generates CodeBlocks project files.
CodeBlocks - Unix Makefiles = Generates CodeBlocks project files.
CodeLite - Ninja = Generates CodeLite project files.
CodeLite - Unix Makefiles = Generates CodeLite project files.
Sublime Text 2 - Ninja = Generates Sublime Text 2 project files.
Sublime Text 2 - Unix Makefiles = Generates Sublime Text 2 project files.
Kate - Ninja = Generates Kate project files.
Kate - Unix Makefiles = Generates Kate project files.
Eclipse CDT4 - Ninja = Generates Eclipse CDT 4.0 project files.
Eclipse CDT4 - Unix Makefiles= Generates Eclipse CDT 4.0 project files.
```

使用此示例，我们将展示为项目切换生成器是多么**EASY**。

### 具体实施

我们将重用前一节示例中的`hello-world.cpp`和`CMakeLists.txt`。惟一的区别在使用CMake时，因为现在必须显式地使用命令行方式，用`-G`切换生成器。

1. 首先，使用以下步骤配置项目:

```bash
$ mkdir -p build
$ cd build
$ cmake -G Ninja ..

-- The CXX compiler identification is GNU 8.1.0
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Configuring done
-- Generating done
-- Build files have been written to: /home/user/cmake-cookbook/chapter-01/recipe-02/cxx-exampl
```

第二步，构建项目：

```bash
$ cmake --build .

[2/2] Linking CXX executable hello-world
```

### 如何工作

与前一个配置相比，每一步的输出没什么变化。每个生成器都有自己的文件集，所以编译步骤的输出和构建目录的内容是不同的：

- `build.ninja`和`rules.ninja`：包含Ninja的所有的构建语句和构建规则。
- `CMakeCache.txt`：CMake会在这个文件中进行缓存，与生成器无关。
- `CMakeFiles`：包含由CMake在配置期间生成的临时文件。
- `cmake_install.cmake`：CMake脚本处理安装规则，并在安装时使用。

`cmake --build .`将`ninja`命令封装在一个跨平台的接口中。

### 更多信息

我们将在第13章中讨论可选生成器和交叉编译。

要了解关于生成器的更多信息，CMake官方文档是一个很好的选择: https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html

## 构建和链接静态库和动态库

**NOTE**: *这个示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-03 找到，其中有C++和Fortran示例。该配置在CMake 3.5版(或更高版本)测试有效的，并且已经在GNU/Linux、macOS和Windows上进行了测试。*

项目中会有单个源文件构建的多个可执行文件的可能。项目中有多个源文件，通常分布在不同子目录中。这种实践有助于项目的源代码结构，而且支持模块化、代码重用和关注点分离。同时，这种分离可以简化并加速项目的重新编译。本示例中，我们将展示如何将源代码编译到库中，以及如何链接这些库。

### 准备工作

回看第一个例子，这里并不再为可执行文件提供单个源文件，我们现在将引入一个类，用来包装要打印到屏幕上的消息。更新一下的`hello-world.cpp`:

```cpp
#include "Message.hpp"

#include <cstdlib>
#include <iostream>

int main() {
  Message say_hello("Hello, CMake World!");
  std::cout << say_hello << std::endl;
  
  Message say_goodbye("Goodbye, CMake World");
  std::cout << say_goodbye << std::endl;
  
  return EXIT_SUCCESS;
}
```

`Message`类包装了一个字符串，并提供重载过的`<<`操作，并且包括两个源码文件：`Message.hpp`头文件与`Message.cpp`源文件。`Message.hpp`中的接口包含以下内容：`Message`类包装了一个字符串，并提供重载过的`<<`操作，并且包括两个源码文件：`Message.hpp`头文件与`Message.cpp`源文件。`Message.hpp`中的接口包含以下内容：

```cpp
#pragma once

#include <iosfwd>
#include <string>

class Message {
public:
  Message(const std::string &m) : message_(m) {}
  friend std::ostream &operator<<(std::ostream &os, Message &obj) {
    return obj.printObject(os);
  }
private:
  std::string message_;
  std::ostream &printObject(std::ostream &os);
};
```

`Message.cpp`实现如下：

```cpp
#include "Message.hpp"

#include <iostream>
#include <string>

std::ostream &Message::printObject(std::ostream &os) {
  os << "This is my very nice message: " << std::endl;
  os << message_;
  return os;
}
```

### 具体实施

这里有两个文件需要编译，所以`CMakeLists.txt`必须进行修改。本例中，先把它们编译成一个库，而不是直接编译成可执行文件:

1. 创建目标——静态库。库的名称和源码文件名相同，具体代码如下：

```cmake
add_library(message
  STATIC
    Message.hpp
    Message.cpp
  )
```

2. 创建`hello-world`可执行文件的目标部分不需要修改：

```cmake
add_executable(hello-world hello-world.cpp)
```

3. 最后，将目标库链接到可执行目标：

```cmake
target_link_libraries(hello-world message)
```

4. 对项目进行配置和构建。库编译完成后，将连接到`hello-world`可执行文件中：

```bash
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .

Scanning dependencies of target message
[ 25%] Building CXX object CMakeFiles/message.dir/Message.cpp.o
[ 50%] Linking CXX static library libmessage.a
[ 50%] Built target message
Scanning dependencies of target hello-world
[ 75%] Building CXX object CMakeFiles/hello-world.dir/hello-world.cpp.o
[100%] Linking CXX executable hello-world
[100%] Built target hello-world
```

```bash
$ ./hello-world

This is my very nice message:
Hello, CMake World!
This is my very nice message:
Goodbye, CMake World
```

### 工作原理

本节引入了两个新命令：

- `add_library(message STATIC Message.hpp Message.cpp)`：生成必要的构建指令，将指定的源码编译到库中。`add_library`的第一个参数是目标名。整个`CMakeLists.txt`中，可使用相同的名称来引用库。生成的库的实际名称将由CMake通过在前面添加前缀`lib`和适当的扩展名作为后缀来形成。生成库是根据第二个参数(`STATIC`或`SHARED`)和操作系统确定的。
- `target_link_libraries(hello-world message)`: 将库链接到可执行文件。此命令还确保`hello-world`可执行文件可以正确地依赖于消息库。因此，在消息库链接到`hello-world`可执行文件之前，需要完成消息库的构建。

编译成功后，构建目录包含`libmessage.a`一个静态库(在GNU/Linux上)和`hello-world`可执行文件。

CMake接受其他值作为`add_library`的第二个参数的有效值，我们来看下本书会用到的值：

- **STATIC**：用于创建静态库，即编译文件的打包存档，以便在链接其他目标时使用，例如：可执行文件。
- **SHARED**：用于创建动态库，即可以动态链接，并在运行时加载的库。可以在`CMakeLists.txt`中使用`add_library(message SHARED Message.hpp Message.cpp)`从静态库切换到动态共享对象(DSO)。
- **OBJECT**：可将给定`add_library`的列表中的源码编译到目标文件，不将它们归档到静态库中，也不能将它们链接到共享对象中。如果需要一次性创建静态库和动态库，那么使用对象库尤其有用。我们将在本示例中演示。
- **MODULE**：又为DSO组。与`SHARED`库不同，它们不链接到项目中的任何目标，不过可以进行动态加载。该参数可以用于构建运行时插件。

CMake还能够生成特殊类型的库，这不会在构建系统中产生输出，但是对于组织目标之间的依赖关系，和构建需求非常有用：

- **IMPORTED**：此类库目标表示位于项目外部的库。此类库的主要用途是，对现有依赖项进行构建。因此，`IMPORTED`库将被视为不可变的。我们将在本书的其他章节演示使用`IMPORTED`库的示例。参见: https://cmake.org/cmake/help/latest/manual/cmakebuildsystem.7.html#imported-targets
- **INTERFACE**：与`IMPORTED`库类似。不过，该类型库可变，没有位置信息。它主要用于项目之外的目标构建使用。我们将在本章第5节中演示`INTERFACE`库的示例。参见: https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html#interface-libraries
- **ALIAS**：顾名思义，这种库为项目中已存在的库目标定义别名。不过，不能为`IMPORTED`库选择别名。参见: https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html#alias-libraries

本例中，我们使用`add_library`直接集合了源代码。后面的章节中，我们将使用`target_sources`汇集源码，特别是在第7章。请参见Craig Scott的这篇精彩博文: https://crascit.com/2016/01/31/enhanced-source-file-handling-with-target_sources/ ，其中有对`target_sources`命令的具体使用。

### 更多信息

现在展示`OBJECT`库的使用，修改`CMakeLists.txt`，如下：

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(recipe-03 LANGUAGES CXX)

add_library(message-objs
	OBJECT
		Message.hpp
		Message.cpp
	)
	
# this is only needed for older compilers
# but doesn't hurt either to have it
set_target_properties(message-objs
	PROPERTIES
		POSITION_INDEPENDENT_CODE 1
	)
	
add_library(message-shared
	SHARED
		$<TARGET_OBJECTS:message-objs>
	)
	
add_library(message-static
	STATIC
		$<TARGET_OBJECTS:message-objs>
	)
	
add_executable(hello-world hello-world.cpp)

target_link_libraries(hello-world message-static)

```

首先，`add_library`改为`add_library(Message-objs OBJECT Message.hpp Message.cpp)`。此外，需要保证编译的目标文件与生成位置无关。可以通过使用`set_target_properties`命令，设置`message-objs`目标的相应属性来实现。

**NOTE**: *可能在某些平台和/或使用较老的编译器上，需要显式地为目标设置`POSITION_INDEPENDENT_CODE`属性。*

现在，可以使用这个对象库来获取静态库(`message-static`)和动态库(`message-shared`)。要注意引用对象库的生成器表达式语法:`$<TARGET_OBJECTS:message-objs>`。生成器表达式是CMake在生成时(即配置之后)构造，用于生成特定于配置的构建输出。参见: https://cmake.org/cmake/help/latest/manual/cmake-generator-expressions.7.html 。我们将在第5章中深入研究生成器表达式。最后，将`hello-world`可执行文件链接到消息库的静态版本。

是否可以让CMake生成同名的两个库？换句话说，它们都可以被称为`message`，而不是`message-static`和`message-share`d吗？我们需要修改这两个目标的属性：

```cmake
add_library(message-shared
  SHARED
    $<TARGET_OBJECTS:message-objs>
	)

set_target_properties(message-shared
	PROPERTIES
		OUTPUT_NAME "message"
	)
	
add_library(message-static
	STATIC
		$<TARGET_OBJECTS:message-objs>
	)
	
set_target_properties(message-static
	PROPERTIES
		OUTPUT_NAME "message"
	)
```

我们可以链接到DSO吗？这取决于操作系统和编译器：

1. GNU/Linux和macOS上，不管选择什么编译器，它都可以工作。
2. Windows上，不能与Visual Studio兼容，但可以与MinGW和MSYS2兼容。

这是为什么呢？生成好的DSO组需要程序员限制符号的可见性。需要在编译器的帮助下实现，但不同的操作系统和编译器上，约定不同。CMake有一个机制来处理这个问题，我们将在第10章中解释它如何工作。

## 用条件句控制编译

**NOTE**:*这个示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-04 找到，其中有一个C++示例。该配置在CMake 3.5版(或更高版本)测试有效的，并且已经在GNU/Linux、macOS和Windows上进行了测试。*

目前为止，看到的示例比较简单，CMake执行流是线性的：从一组源文件到单个可执行文件，也可以生成静态库或动态库。为了确保完全控制构建项目、配置、编译和链接所涉及的所有步骤的执行流，CMake提供了自己的语言。本节中，我们将探索条件结构`if-else- else-endif`的使用。

**NOTE**: *CMake语言相当庞杂，由基本的控制结构、特定于CMake的命令和使用新函数模块化扩展语言的基础设施组成。完整的概览可以在这里找到: https://cmake.org/cmake/help/latest/manual/cmake-language.7.html*

### 具体实施

从与上一个示例的的源代码开始，我们希望能够在不同的两种行为之间进行切换：

1. 将`Message.hpp`和`Message.cpp`构建成一个库(静态或动态)，然后将生成库链接到`hello-world`可执行文件中。
2. 将`Message.hpp`，`Message.cpp`和`hello-world.cpp`构建成一个可执行文件，但不生成任何一个库。

让我们来看看如何使用`CMakeLists.txt`来实现：

1. 首先，定义最低CMake版本、项目名称和支持的语言：

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(recipe-04 LANGUAGES CXX)
```

2. 我们引入了一个新变量`USE_LIBRARY`，这是一个逻辑变量，值为`OFF`。我们还打印了它的值：

```cmake
set(USE_LIBRARY OFF)

message(STATUS "Compile sources into a library? ${USE_LIBRARY}")
```

3. CMake中定义`BUILD_SHARED_LIBS`全局变量，并设置为`OFF`。调用`add_library`并省略第二个参数，将构建一个静态库：

```cmake
set(BUILD_SHARED_LIBS OFF)
```

4. 然后，引入一个变量`_sources`，包括`Message.hpp`和`Message.cpp`：

```cmake
list(APPEND _sources Message.hpp Message.cpp)
```

5. 然后，引入一个基于`USE_LIBRARY`值的`if-else`语句。如果逻辑为真，则`Message.hpp`和`Message.cpp`将打包成一个库：

```cmake
if(USE_LIBRARY)
	# add_library will create a static library
	# since BUILD_SHARED_LIBS is OFF
	add_library(message ${_sources})
	add_executable(hello-world hello-world.cpp)
	target_link_libraries(hello-world message)
else()
	add_executable(hello-world hello-world.cpp ${_sources})
endif()
```

1. 我们可以再次使用相同的命令集进行构建。由于`USE_LIBRARY`为`OFF`, `hello-world`可执行文件将使用所有源文件来编译。可以通过在GNU/Linux上，运行`objdump -x`命令进行验证。

### 工作原理

我们介绍了两个变量：`USE_LIBRARY`和`BUILD_SHARED_LIBS`。这两个变量都设置为`OFF`。如CMake语言文档中描述，逻辑真或假可以用多种方式表示：

- 如果将逻辑变量设置为以下任意一种：`1`、`ON`、`YES`、`true`、`Y`或非零数，则逻辑变量为`true`。
- 如果将逻辑变量设置为以下任意一种：`0`、`OFF`、`NO`、`false`、`N`、`IGNORE、NOTFOUND`、空字符串，或者以`-NOTFOUND`为后缀，则逻辑变量为`false`。

`USE_LIBRARY`变量将在第一个和第二个行为之间切换。`BUILD_SHARED_LIBS`是CMake的一个全局标志。因为CMake内部要查询`BUILD_SHARED_LIBS`全局变量，所以`add_library`命令可以在不传递`STATIC/SHARED/OBJECT`参数的情况下调用；如果为`false`或未定义，将生成一个静态库。

这个例子说明，可以引入条件来控制CMake中的执行流。但是，当前的设置不允许从外部切换，不需要手动修改`CMakeLists.txt`。原则上，我们希望能够向用户开放所有设置，这样就可以在不修改构建代码的情况下调整配置，稍后将展示如何做到这一点。

**NOTE**:*`else()`和`endif()`中的`()`，可能会让刚开始学习CMake代码的同学感到惊讶。其历史原因是，因为其能够指出指令的作用范围。例如，可以使用`if(USE_LIBRARY)…else(USE_LIBRARY)…endif(USE_LIBIRAY)`。这个格式并不唯一，可以根据个人喜好来决定使用哪种格式。*

**TIPS**:*`_sources`变量是一个局部变量，不应该在当前范围之外使用，可以在名称前加下划线。*

## 向用户显示选项

**NOTE**: *这个示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-05 找到，其中有一个C++示例。该配置在CMake 3.5版(或更高版本)测试有效的，并且已经在GNU/Linux、macOS和Windows上进行了测试。*

前面的配置中，我们引入了条件句：通过硬编码的方式给定逻辑变量值。不过，这会影响用户修改这些变量。CMake代码没有向读者传达，该值可以从外部进行修改。推荐在`CMakeLists.txt`中使用`option()`命令，以选项的形式显示逻辑开关，用于外部设置，从而切换构建系统的生成行为。本节的示例将向您展示，如何使用这个命令。

### 具体实施

看一下前面示例中的静态/动态库示例。与其硬编码`USE_LIBRARY`为`ON`或`OFF`，现在为其设置一个默认值，同时也可以从外部进行更改：

1. 用一个选项替换上一个示例的`set(USE_LIBRARY OFF)`命令。该选项将修改`USE_LIBRARY`的值，并设置其默认值为`OFF`：

```cmake
option(USE_LIBRARY "Compile sources into a library" OFF)
```

2. 现在，可以通过CMake的`-D`CLI选项，将信息传递给CMake来切换库的行为：

```bash
$ mkdir -p build
$ cd build
$ cmake -D USE_LIBRARY=ON ..

-- ...
-- Compile sources into a library? ON
-- ...

$ cmake --build .

Scanning dependencies of target message
[ 25%] Building CXX object CMakeFiles/message.dir/Message.cpp.o
[ 50%] Linking CXX static library libmessage.a
[ 50%] Built target message
Scanning dependencies of target hello-world
[ 75%] Building CXX object CMakeFiles/hello-world.dir/hello-world.cpp.o
[100%] Linking CXX executable hello-world
[100%] Built target hello-world
```

`-D`开关用于为CMake设置任何类型的变量：逻辑变量、路径等等。

### 工作原理

`option`可接受三个参数：

```cmake
option(<option_variable> "help string" [initial value])
```

- `<option_variable>`表示该选项的变量的名称。
- `"help string"`记录选项的字符串，在CMake的终端或图形用户界面中可见。
- `[initial value]`选项的默认值，可以是`ON`或`OFF`。

### 更多信息

有时选项之间会有依赖的情况。示例中，我们提供生成静态库或动态库的选项。但是，如果没有将`USE_LIBRARY`逻辑设置为`ON`，则此选项没有任何意义。CMake提供`cmake_dependent_option()`命令用来定义依赖于其他选项的选项：

```cmake
include(CMakeDependentOption)

# second option depends on the value of the first
cmake_dependent_option(
	MAKE_STATIC_LIBRARY "Compile sources into a static library" OFF
	"USE_LIBRARY" ON
	)
	
# third option depends on the value of the first
cmake_dependent_option(
	MAKE_SHARED_LIBRARY "Compile sources into a shared library" ON
	"USE_LIBRARY" ON
	)
```

如果`USE_LIBRARY`为`ON`，`MAKE_STATIC_LIBRARY`默认值为`OFF`，否则`MAKE_SHARED_LIBRARY`默认值为`ON`。可以这样运行：

```bash
$ cmake -D USE_LIBRARY=OFF -D MAKE_SHARED_LIBRARY=ON ..
```

这仍然不会构建库，因为`USE_LIBRARY`仍然为`OFF`。

CMake有适当的机制，通过包含模块来扩展其语法和功能，这些模块要么是CMake自带的，要么是定制的。本例中，包含了一个名为`CMakeDependentOption`的模块。如果没有`include`这个模块，`cmake_dependent_option()`命令将不可用。参见 https://cmake.org/cmake/help/latest/module/CMakeDependentOption.html

**TIPS**:*手册中的任何模块都可以以命令行的方式使用`cmake --help-module <name-of-module>`。例如，`cmake --help-module CMakeDependentOption`将打印刚才讨论的模块的手册页(帮助页面)。*

## 指定编译器

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-06 中找到，其中有一个C++/C示例。该配置在CMake 3.5版(或更高版本)下测试没问题，并且已经在GNU/Linux、macOS和Windows上进行了测试。*

目前为止，我们还没有过多考虑如何选择编译器。CMake可以根据平台和生成器选择编译器，还能将编译器标志设置为默认值。然而，我们通常控制编译器的选择。在后面的示例中，我们还将考虑构建类型的选择，并展示如何控制编译器标志。

### 具体实施

如何选择一个特定的编译器？例如，如果想使用Intel或Portland Group编译器怎么办？CMake将语言的编译器存储在`CMAKE_<LANG>_COMPILER`变量中，其中`<LANG>`是受支持的任何一种语言，对于我们的目的是`CXX`、`C`或`Fortran`。用户可以通过以下两种方式之一设置此变量：

1. 使用CLI中的`-D`选项，例如：

```bash
$ cmake -D CMAKE_CXX_COMPILER=clang++ ..
```

2. 通过导出环境变量`CXX`(C++编译器)、`CC`(C编译器)和`FC`(Fortran编译器)。例如，使用这个命令使用`clang++`作为`C++`编译器：

```bash
$ env CXX=clang++ cmake ..
```

到目前为止讨论的示例，都可以通过传递适当的选项，配置合适的编译器。

**NOTE**:*CMake了解运行环境，可以通过其CLI的`-D`开关或环境变量设置许多选项。前一种机制覆盖后一种机制，但是我们建议使用`-D`显式设置选项。显式优于隐式，因为环境变量可能被设置为不适合(当前项目)的值。*

我们在这里假设，其他编译器在标准路径中可用，CMake在标准路径中执行查找编译器。如果不是这样，用户将需要将完整的编译器可执行文件或包装器路径传递给CMake。

**TIPS**:*我们建议使用`-D CMAKE_<LANG>_COMPILER`CLI选项设置编译器，而不是导出`CXX`、`CC`和`FC`。这是确保跨平台并与非POSIX兼容的唯一方法。为了避免变量污染环境，这些变量可能会影响与项目一起构建的外部库环境。*

### 作原理

配置时，CMake会进行一系列平台测试，以确定哪些编译器可用，以及它们是否适合当前的项目。一个合适的编译器不仅取决于我们所使用的平台，还取决于我们想要使用的生成器。CMake执行的第一个测试基于项目语言的编译器的名称。例如，`cc`是一个工作的`C`编译器，那么它将用作`C`项目的默认编译器。GNU/Linux上，使用Unix Makefile或Ninja时, GCC家族中的编译器很可能是`C++`、`C`和`Fortran`的默认选择。Microsoft Windows上，将选择Visual Studio中的`C++`和`C`编译器(前提是Visual Studio是生成器)。如果选择MinGW或MSYS Makefile作为生成器，则默认使用MinGW编译器。

### 更多信息

我们的平台上的CMake，在哪里可以找到可用的编译器和编译器标志？CMake提供`--system-information`标志，它将把关于系统的所有信息转储到屏幕或文件中。要查看这个信息，请尝试以下操作：

```bash
$ cmake --system-information information.txt
```

文件中(本例中是`information.txt`)可以看到`CMAKE_CXX_COMPILER`、`CMAKE_C_COMPILER`和`CMAKE_Fortran_COMPILER`的默认值，以及默认标志。我们将在下一个示例中看到相关的标志。

CMake提供了额外的变量来与编译器交互：

- `CMAKE_<LANG>_COMPILER_LOADED`:如果为项目启用了语言`<LANG>`，则将设置为`TRUE`。
- `CMAKE_<LANG>_COMPILER_ID`:编译器标识字符串，编译器供应商所特有。例如，`GCC`用于GNU编译器集合，`AppleClang`用于macOS上的Clang, `MSVC`用于Microsoft Visual Studio编译器。注意，不能保证为所有编译器或语言定义此变量。
- `CMAKE_COMPILER_IS_GNU<LANG>`:如果语言`<LANG>`是GNU编译器集合的一部分，则将此逻辑变量设置为`TRUE`。注意变量名的`<LANG>`部分遵循GNU约定：C语言为`CC`, C++语言为`CXX`, Fortran语言为`G77`。
- `CMAKE_<LANG>_COMPILER_VERSION`:此变量包含一个字符串，该字符串给定语言的编译器版本。版本信息在`major[.minor[.patch[.tweak]]]`中给出。但是，对于`CMAKE_<LANG>_COMPILER_ID`，不能保证所有编译器或语言都定义了此变量。

我们可以尝试使用不同的编译器，配置下面的示例`CMakeLists.txt`。这个例子中，我们将使用CMake变量来探索已使用的编译器(及版本)：

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(recipe-06 LANGUAGES C CXX)

message(STATUS "Is the C++ compiler loaded? ${CMAKE_CXX_COMPILER_LOADED}")
if(CMAKE_CXX_COMPILER_LOADED)
	message(STATUS "The C++ compiler ID is: ${CMAKE_CXX_COMPILER_ID}")
	message(STATUS "Is the C++ from GNU? ${CMAKE_COMPILER_IS_GNUCXX}")
	message(STATUS "The C++ compiler version is: ${CMAKE_CXX_COMPILER_VERSION}")
endif()

message(STATUS "Is the C compiler loaded? ${CMAKE_C_COMPILER_LOADED}")
if(CMAKE_C_COMPILER_LOADED)
	message(STATUS "The C compiler ID is: ${CMAKE_C_COMPILER_ID}")
	message(STATUS "Is the C from GNU? ${CMAKE_COMPILER_IS_GNUCC}")
	message(STATUS "The C compiler version is: ${CMAKE_C_COMPILER_VERSION}")
endif()
```

注意，这个例子不包含任何目标，没有要构建的东西，我们只关注配置步骤:

```bash
$ mkdir -p build
$ cd build
$ cmake ..

...
-- Is the C++ compiler loaded? 1
-- The C++ compiler ID is: GNU
-- Is the C++ from GNU? 1
-- The C++ compiler version is: 8.1.0
-- Is the C compiler loaded? 1
-- The C compiler ID is: GNU
-- Is the C from GNU? 1
-- The C compiler version is: 8.1.0
...
```

当然，输出将取决于可用和已选择的编译器(及版本)。

## 切换构建类型

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-07 中找到，包含一个C++/C示例。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

CMake可以配置构建类型，例如：Debug、Release等。配置时，可以为Debug或Release构建设置相关的选项或属性，例如：编译器和链接器标志。控制生成构建系统使用的配置变量是`CMAKE_BUILD_TYPE`。该变量默认为空，CMake识别的值为:

1. **Debug**：用于在没有优化的情况下，使用带有调试符号构建库或可执行文件。
2. **Release**：用于构建的优化的库或可执行文件，不包含调试符号。
3. **RelWithDebInfo**：用于构建较少的优化库或可执行文件，包含调试符号。
4. **MinSizeRel**：用于不增加目标代码大小的优化方式，来构建库或可执行文件。

### 具体实施

示例中，我们将展示如何为项目设置构建类型：

1. 首先，定义最低CMake版本、项目名称和支持的语言：

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(recipe-07 LANGUAGES C CXX)
```

2. 然后，设置一个默认的构建类型(本例中是Release)，并打印一条消息。要注意的是，该变量被设置为缓存变量，可以通过缓存进行编辑：

```cmake
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
```

3. 最后，打印出CMake设置的相应编译标志：

```cmake
message(STATUS "C flags, Debug configuration: ${CMAKE_C_FLAGS_DEBUG}")
message(STATUS "C flags, Release configuration: ${CMAKE_C_FLAGS_RELEASE}")
message(STATUS "C flags, Release configuration with Debug info: ${CMAKE_C_FLAGS_RELWITHDEBINFO}")
message(STATUS "C flags, minimal Release configuration: ${CMAKE_C_FLAGS_MINSIZEREL}")
message(STATUS "C++ flags, Debug configuration: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "C++ flags, Release configuration: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "C++ flags, Release configuration with Debug info: ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message(STATUS "C++ flags, minimal Release configuration: ${CMAKE_CXX_FLAGS_MINSIZEREL}")
```

4. 验证配置的输出:

```bash
$ mkdir -p build
$ cd build
$ cmake ..

...
-- Build type: Release
-- C flags, Debug configuration: -g
-- C flags, Release configuration: -O3 -DNDEBUG
-- C flags, Release configuration with Debug info: -O2 -g -DNDEBUG
-- C flags, minimal Release configuration: -Os -DNDEBUG
-- C++ flags, Debug configuration: -g
-- C++ flags, Release configuration: -O3 -DNDEBUG
-- C++ flags, Release configuration with Debug info: -O2 -g -DNDEBUG
-- C++ flags, minimal Release configuration: -Os -DNDEBUG
```

5. 切换构建类型:

```bash
$ cmake -D CMAKE_BUILD_TYPE=Debug ..

-- Build type: Debug
-- C flags, Debug configuration: -g
-- C flags, Release configuration: -O3 -DNDEBUG
-- C flags, Release configuration with Debug info: -O2 -g -DNDEBUG
-- C flags, minimal Release configuration: -Os -DNDEBUG
-- C++ flags, Debug configuration: -g
-- C++ flags, Release configuration: -O3 -DNDEBUG
-- C++ flags, Release configuration with Debug info: -O2 -g -DNDEBUG
-- C++ flags, minimal Release configuration: -Os -DNDEBUG
```

### 工作原理

我们演示了如何设置默认构建类型，以及如何(从命令行)覆盖它。这样，就可以控制项目，是使用优化，还是关闭优化启用调试。我们还看到了不同配置使用了哪些标志，这主要取决于选择的编译器。需要在运行CMake时显式地打印标志，也可以仔细阅读运行`CMake --system-information`的输出，以了解当前平台、默认编译器和语言的默认组合是什么。下一个示例中，我们将讨论如何为不同的编译器和不同的构建类型，扩展或调整编译器标志。

### 更多信息

我们展示了变量`CMAKE_BUILD_TYPE`，如何切换生成构建系统的配置(这个链接中有说明: https://cmake.org/cmake/help/v3.5/variable/CMAKE_BUILD_TYPE.html )。Release和Debug配置中构建项目通常很有用，例如：评估编译器优化级别的效果。对于单配置生成器，如Unix Makefile、MSYS Makefile或Ninja，因为要对项目重新配置，这里需要运行CMake两次。不过，CMake也支持复合配置生成器。这些通常是集成开发环境提供的项目文件，最显著的是Visual Studio和Xcode，它们可以同时处理多个配置。可以使用`CMAKE_CONFIGURATION_TYPES`变量可以对这些生成器的可用配置类型进行调整，该变量将接受一个值列表(可从这个链接获得文档:https://cmake.org/cmake/help/v3.5/variable/CMAKE_CONFIGURATION_TYPES.html)。

下面是对Visual Studio的CMake调用:

```bash
$ mkdir -p build
$ cd build
$ cmake .. -G"Visual Studio 12 2017 Win64" -D CMAKE_CONFIGURATION_TYPES="Release;Debug"
```

将为Release和Debug配置生成一个构建树。然后，您可以使`--config`标志来决定构建这两个中的哪一个:

```bash
$ cmake --build . --config Release
```

**NOTE**:*当使用单配置生成器开发代码时，为Release版和Debug创建单独的构建目录，两者使用相同的源代码。这样，就可以在两者之间切换，而不用重新配置和编译。*

## 设置编译器选项

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-08 中找到，有一个C++示例。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

前面的示例展示了如何探测CMake，从而获得关于编译器的信息，以及如何切换项目中的编译器。后一个任务是控制项目的编译器标志。CMake为调整或扩展编译器标志提供了很大的灵活性，您可以选择下面两种方法:

- CMake将编译选项视为目标属性。因此，可以根据每个目标设置编译选项，而不需要覆盖CMake默认值。
- 可以使用`-D`CLI标志直接修改`CMAKE_<LANG>_FLAGS_<CONFIG>`变量。这将影响项目中的所有目标，并覆盖或扩展CMake默认值。

本示例中，我们将展示这两种方法。

### 准备工作

编写一个示例程序，计算不同几何形状的面积，`computer_area.cpp`：

```cpp
#include "geometry_circle.hpp"
#include "geometry_polygon.hpp"
#include "geometry_rhombus.hpp"
#include "geometry_square.hpp"

#include <cstdlib>
#include <iostream>

int main() {
  using namespace geometry;
  
  double radius = 2.5293;
  double A_circle = area::circle(radius);
  std::cout << "A circle of radius " << radius << " has an area of " << A_circle
            << std::endl;
  
  int nSides = 19;
  double side = 1.29312;
  double A_polygon = area::polygon(nSides, side);
  std::cout << "A regular polygon of " << nSides << " sides of length " << side
            << " has an area of " << A_polygon << std::endl;

  double d1 = 5.0;
  double d2 = 7.8912;
  double A_rhombus = area::rhombus(d1, d2);
  std::cout << "A rhombus of major diagonal " << d1 << " and minor diagonal " << d2
            << " has an area of " << A_rhombus << std::endl;
  
  double l = 10.0;
  double A_square = area::square(l);
  std::cout << "A square of side " << l << " has an area of " << A_square
  << std::endl;

  return EXIT_SUCCESS;
}
```

函数的各种实现分布在不同的文件中，每个几何形状都有一个头文件和源文件。总共有4个头文件和5个源文件要编译：

```bash
.
├─ CMakeLists.txt
├─ compute-areas.cpp
├─ geometry_circle.cpp
├─ geometry_circle.hpp
├─ geometry_polygon.cpp
├─ geometry_polygon.hpp
├─ geometry_rhombus.cpp
├─ geometry_rhombus.hpp
├─ geometry_square.cpp
└─ geometry_square.hpp
```

我们不会为所有文件提供清单，读者可以参考 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-08 。

### 具体实施

现在已经有了源代码，我们的目标是配置项目，并使用编译器标示进行实验:

1. 设置CMake的最低版本:

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
```

2. 声明项目名称和语言:

```c make
project(recipe-08 LANGUAGES CXX)
```

3. 然后，打印当前编译器标志。CMake将对所有C++目标使用这些:

```cmake
message("C++ compiler flags: ${CMAKE_CXX_FLAGS}")
```

4. 为目标准备了标志列表，其中一些将无法在Windows上使用:

```cmake
list(APPEND flags "-fPIC" "-Wall")
if(NOT WIN32)
  list(APPEND flags "-Wextra" "-Wpedantic")
endif()
```

5. 添加了一个新的目标——`geometry`库，并列出它的源依赖关系:

```cmake
add_library(geometry
  STATIC
    geometry_circle.cpp
    geometry_circle.hpp
    geometry_polygon.cpp
    geometry_polygon.hpp
    geometry_rhombus.cpp
    geometry_rhombus.hpp
    geometry_square.cpp
    geometry_square.hpp
  )
```

6. 为这个库目标设置了编译选项:

```cmake
target_compile_options(geometry
  PRIVATE
    ${flags}
  )
```

7. 然后，将生成`compute-areas`可执行文件作为一个目标:

```cmake
add_executable(compute-areas compute-areas.cpp)
```

8. 还为可执行目标设置了编译选项:

```cmake
target_compile_options(compute-areas
  PRIVATE
    "-fPIC"
  )
```

9. 最后，将可执行文件链接到geometry库:

```cmake
target_link_libraries(compute-areas geometry)
```

### 如何工作

本例中，警告标志有`-Wall`、`-Wextra`和`-Wpedantic`，将这些标示添加到`geometry`目标的编译选项中； `compute-areas`和 `geometry`目标都将使用`-fPIC`标志。编译选项可以添加三个级别的可见性：`INTERFACE`、`PUBLIC`和`PRIVATE`。

可见性的含义如下:

- **PRIVATE**，编译选项会应用于给定的目标，不会传递给与目标相关的目标。我们的示例中， 即使`compute-areas`将链接到`geometry`库，`compute-areas`也不会继承`geometry`目标上设置的编译器选项。
- **INTERFACE**，给定的编译选项将只应用于指定目标，并传递给与目标相关的目标。
- **PUBLIC**，编译选项将应用于指定目标和使用它的目标。

目标属性的可见性CMake的核心，我们将在本书中经常讨论这个话题。以这种方式添加编译选项，不会影响全局CMake变量`CMAKE_<LANG>_FLAGS_<CONFIG>`，并能更细粒度控制在哪些目标上使用哪些选项。

我们如何验证，这些标志是否按照我们的意图正确使用呢？或者换句话说，如何确定项目在CMake构建时，实际使用了哪些编译标志？一种方法是，使用CMake将额外的参数传递给本地构建工具。本例中会设置环境变量`VERBOSE=1`：

```bash
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build . -- VERBOSE=1

... lots of output ...

[ 14%] Building CXX object CMakeFiles/geometry.dir/geometry_circle.cpp.o
/usr/bin/c++ -fPIC -Wall -Wextra -Wpedantic -o CMakeFiles/geometry.dir/geometry_circle.cpp.o -c /home/bast/tmp/cmake-cookbook/chapter-01/recipe-08/cxx-example/geometry_circle.cpp
[ 28%] Building CXX object CMakeFiles/geometry.dir/geometry_polygon.cpp.o
/usr/bin/c++ -fPIC -Wall -Wextra -Wpedantic -o CMakeFiles/geometry.dir/geometry_polygon.cpp.o -c /home/bast/tmp/cmake-cookbook/chapter-01/recipe-08/cxx-example/geometry_polygon.cpp
[ 42%] Building CXX object CMakeFiles/geometry.dir/geometry_rhombus.cpp.o
/usr/bin/c++ -fPIC -Wall -Wextra -Wpedantic -o CMakeFiles/geometry.dir/geometry_rhombus.cpp.o -c /home/bast/tmp/cmake-cookbook/chapter-01/recipe-08/cxx-example/geometry_rhombus.cpp
[ 57%] Building CXX object CMakeFiles/geometry.dir/geometry_square.cpp.o
/usr/bin/c++ -fPIC -Wall -Wextra -Wpedantic -o CMakeFiles/geometry.dir/geometry_square.cpp.o -c /home/bast/tmp/cmake-cookbook/chapter-01/recipe-08/cxx-example/geometry_square.cpp

... more output ...

[ 85%] Building CXX object CMakeFiles/compute-areas.dir/compute-areas.cpp.o
/usr/bin/c++ -fPIC -o CMakeFiles/compute-areas.dir/compute-areas.cpp.o -c /home/bast/tmp/cmake-cookbook/chapter-01/recipe-08/cxx-example/compute-areas.cpp

... more output ...
```

输出确认编译标志，确认指令设置正确。

控制编译器标志的第二种方法，不用对`CMakeLists.txt`进行修改。如果想在这个项目中修改`geometry`和`compute-areas`目标的编译器选项，可以使用CMake参数进行配置：

```bash
$ cmake -D CMAKE_CXX_FLAGS="-fno-exceptions -fno-rtti" ..
```

这个命令将编译项目，禁用异常和运行时类型标识(RTTI)。

也可以使用全局标志，可以使用`CMakeLists.txt`运行以下命令：

```bash
$ cmake -D CMAKE_CXX_FLAGS="-fno-exceptions -fno-rtti" ..
```

这将使用`-fno-rtti - fpic - wall - Wextra - wpedantic`配置`geometry`目标，同时使用`-fno exception -fno-rtti - fpic`配置`compute-areas`。

**NOTE**:*本书中，我们推荐为每个目标设置编译器标志。使用`target_compile_options()`不仅允许对编译选项进行细粒度控制，而且还可以更好地与CMake的更高级特性进行集成。*

### 更多信息

大多数时候，编译器有特性标示。当前的例子只适用于`GCC`和`Clang`；其他供应商的编译器不确定是否会理解(如果不是全部)这些标志。如果项目是真正跨平台，那么这个问题就必须得到解决，有三种方法可以解决这个问题。

最典型的方法是将所需编译器标志列表附加到每个配置类型CMake变量`CMAKE_<LANG>_FLAGS_<CONFIG>`。标志确定设置为给定编译器有效的标志，因此将包含在`if-endif`子句中，用于检查`CMAKE_<LANG>_COMPILER_ID`变量，例如：

```cmake
if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
  list(APPEND CMAKE_CXX_FLAGS "-fno-rtti" "-fno-exceptions")
  list(APPEND CMAKE_CXX_FLAGS_DEBUG "-Wsuggest-final-types" "-Wsuggest-final-methods" "-Wsuggest-override")
  list(APPEND CMAKE_CXX_FLAGS_RELEASE "-O3" "-Wno-unused")
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES Clang)
  list(APPEND CMAKE_CXX_FLAGS "-fno-rtti" "-fno-exceptions" "-Qunused-arguments" "-fcolor-diagnostics")
  list(APPEND CMAKE_CXX_FLAGS_DEBUG "-Wdocumentation")
  list(APPEND CMAKE_CXX_FLAGS_RELEASE "-O3" "-Wno-unused")
endif()
```

更细粒度的方法是，不修改`CMAKE_<LANG>_FLAGS_<CONFIG>`变量，而是定义特定的标志列表：

```cmake
set(COMPILER_FLAGS)
set(COMPILER_FLAGS_DEBUG)
set(COMPILER_FLAGS_RELEASE)

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
  list(APPEND CXX_FLAGS "-fno-rtti" "-fno-exceptions")
  list(APPEND CXX_FLAGS_DEBUG "-Wsuggest-final-types" "-Wsuggest-final-methods" "-Wsuggest-override")
  list(APPEND CXX_FLAGS_RELEASE "-O3" "-Wno-unused")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES Clang)
  list(APPEND CXX_FLAGS "-fno-rtti" "-fno-exceptions" "-Qunused-arguments" "-fcolor-diagnostics")
  list(APPEND CXX_FLAGS_DEBUG "-Wdocumentation")
  list(APPEND CXX_FLAGS_RELEASE "-O3" "-Wno-unused")
endif()
```

稍后，使用生成器表达式来设置编译器标志的基础上，为每个配置和每个目标生成构建系统:

```cmake
target_compile_option(compute-areas
  PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
  )
```

当前示例中展示了这两种方法，我们推荐后者(特定于项目的变量和`target_compile_options`)。

两种方法都有效，并在许多项目中得到广泛应用。不过，每种方式都有缺点。`CMAKE_<LANG>_COMPILER_ID`不能保证为所有编译器都定义。此外，一些标志可能会被弃用，或者在编译器的较晚版本中引入。与`CMAKE_<LANG>_COMPILER_ID`类似，`CMAKE_<LANG>_COMPILER_VERSION`变量不能保证为所有语言和供应商都提供定义。尽管检查这些变量的方式非常流行，但我们认为更健壮的替代方法是检查所需的标志集是否与给定的编译器一起工作，这样项目中实际上只使用有效的标志。结合特定于项目的变量、`target_compile_options`和生成器表达式，会让解决方案变得非常强大。我们将在第7章的第3节中展示，如何使用`check-and-set`模式。

## 为语言设定标准

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-09 中找到，包含一个C++和Fortran示例。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

编程语言有不同的标准，即提供改进的语言版本。启用新标准是通过设置适当的编译器标志来实现的。前面的示例中，我们已经展示了如何为每个目标或全局进行配置。3.1版本中，CMake引入了一个独立于平台和编译器的机制，用于为`C++`和`C`设置语言标准：为目标设置`<LANG>_STANDARD`属性。

### 准备工作

对于下面的示例，需要一个符合`C++14`标准或更高版本的`C++`编译器。此示例代码定义了动物的多态，我们使用`std::unique_ptr`作为结构中的基类：

```cpp
std::unique_ptr<Animal> cat = Cat("Simon");
std::unique_ptr<Animal> dog = Dog("Marlowe);
```

没有为各种子类型显式地使用构造函数，而是使用工厂方法的实现。工厂方法使用`C++11`的可变参数模板实现。它包含继承层次结构中每个对象的创建函数映射：

```cpp
typedef std::function<std::unique_ptr<Animal>(const
std::string &)> CreateAnimal;
```

基于预先分配的标签来分派它们，创建对象:

```cpp
std::unique_ptr<Animal> simon = farm.create("CAT", "Simon");
std::unique_ptr<Animal> marlowe = farm.create("DOG", "Marlowe");
```

标签和创建功能在工厂使用前已注册:

```cpp
Factory<CreateAnimal> farm;
farm.subscribe("CAT", [](const std::string & n) { return std::make_unique<Cat>(n); });
farm.subscribe("DOG", [](const std::string & n) { return std::make_unique<Dog>(n); });
```

使用`C++11 Lambda`函数定义创建函数，使用`std::make_unique`来避免引入裸指针的操作。这个工厂函数是在`C++14`中引入。

**NOTE**:*CMake的这一功能是在3.1版中添加的，并且还在更新。CMake的后续版本为`C++`标准的后续版本和不同的编译器，提供了越来越好的支持。我们建议您在文档页面上检查您选择的编译器是否受支持: https://cmake.org/cmake/help/latest/manual/cmake-compile-features.7.html#supported-compiler*

### 具体实施

将逐步构建`CMakeLists.txt`，并展示如何设置语言标准(本例中是`C++14`):

1. 声明最低要求的CMake版本，项目名称和语言:

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(recipe-09 LANGUAGES CXX)
```

2. 要求在Windows上导出所有库符号:

```cmake
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
```

3. 需要为库添加一个目标，这将编译源代码为一个动态库:

```cmake
add_library(animals
  SHARED
    Animal.cpp
    Animal.hpp
    Cat.cpp
    Cat.hpp
    Dog.cpp
    Dog.hpp
    Factory.hpp
  )
```

4. 现在，为目标设置了`CXX_STANDARD`、`CXX_EXTENSIONS`和`CXX_STANDARD_REQUIRED`属性。还设置了`position_independent ent_code`属性，以避免在使用一些编译器构建DSO时出现问题:

```cmake
set_target_properties(animals
  PROPERTIES
    CXX_STANDARD 14
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE 1
)
```

5. 然后，为”动物农场”的可执行文件添加一个新目标，并设置它的属性:

```cmake
add_executable(animal-farm animal-farm.cpp)
set_target_properties(animal-farm
  PROPERTIES
    CXX_STANDARD 14
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON
  )
```

6. 最后，将可执行文件链接到库:

```cmake
target_link_libraries(animal-farm animals)
```

7. 现在，来看看猫和狗都说了什么:

```bash
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
$ ./animal-farm

I'm Simon the cat!
I'm Marlowe the dog!
```

### 工作原理

步骤4和步骤5中，我们为动物和动物农场目标设置了一些属性:

- **CXX_STANDARD**会设置我们想要的标准。
- **CXX_EXTENSIONS**告诉CMake，只启用`ISO C++`标准的编译器标志，而不使用特定编译器的扩展。
- **CXX_STANDARD_REQUIRED**指定所选标准的版本。如果这个版本不可用，CMake将停止配置并出现错误。当这个属性被设置为`OFF`时，CMake将寻找下一个标准的最新版本，直到一个合适的标志。这意味着，首先查找`C++14`，然后是`C++11`，然后是`C++98`。（译者注：目前会从`C++20`或`C++17`开始查找）

**NOTE**:*本书发布时，还没有`Fortran_STANDARD`可用，但是可以使用`target_compile_options`设置标准，可以参见: https://github.com/devcafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-09*

**TIPS**:*如果语言标准是所有目标共享的全局属性，那么可以将`CMAKE_<LANG>_STANDARD`、`CMAKE_<LANG>_EXTENSIONS`和`CMAKE_<LANG>_STANDARD_REQUIRED`变量设置为相应的值。所有目标上的对应属性都将使用这些设置。*

### 更多信息

通过引入编译特性，CMake对语言标准提供了更精细的控制。这些是语言标准引入的特性，比如`C++11`中的可变参数模板和`Lambda`表达式，以及`C++14`中的自动返回类型推断。可以使用`target_compile_features()`命令要求为特定的目标提供特定的特性，CMake将自动为标准设置正确的编译器标志。也可以让CMake为可选编译器特性，生成兼容头文件。

**TIPS**:*我们建议阅读CMake在线文档，全面了解`cmake-compile-features`和如何处理编译特性和语言标准: https://cmake.org/cmake/help/latest/manual/cmake-compile-features.7.html 。*

## 使用控制流

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-01/recipe-10 中找到，有一个C++示例。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

本章前面的示例中，已经使用过`if-else-endif`。CMake还提供了创建循环的语言工具：`foreach endforeach`和`while-endwhile`。两者都可以与`break`结合使用，以便尽早从循环中跳出。本示例将展示如何使用`foreach`，来循环源文件列表。我们将应用这样的循环，在引入新目标的前提下，来为一组源文件进行优化降级。

### 准备工作

将重用第8节中的几何示例，目标是通过将一些源代码汇集到一个列表中，从而微调编译器的优化。

### 具体实施

下面是`CMakeLists.txt`中要的详细步骤:

1. 与示例8中一样，指定了CMake的最低版本、项目名称和语言，并声明了几何库目标:



```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(recipe-10 LANGUAGES CXX)
add_library(geometry
  STATIC
    geometry_circle.cpp
    geometry_circle.hpp
    geometry_polygon.cpp
    geometry_polygon.hpp
    geometry_rhombus.cpp
    geometry_rhombus.hpp
    geometry_square.cpp
    geometry_square.hpp
  )
```

2. 使用`-O3`编译器优化级别编译库，对目标设置一个私有编译器选项:

```cmake
target_compile_options(geometry
  PRIVATE
  	-O3
  )
```

3. 然后，生成一个源文件列表，以较低的优化选项进行编译:

```cmake
list(
  APPEND sources_with_lower_optimization
    geometry_circle.cpp
    geometry_rhombus.cpp
  )
```

4. 循环这些源文件，将它们的优化级别调到`-O2`。使用它们的源文件属性完成:

```cmake
message(STATUS "Setting source properties using IN LISTS syntax:")
foreach(_source IN LISTS sources_with_lower_optimization)
  set_source_files_properties(${_source} PROPERTIES COMPILE_FLAGS -O2)
  message(STATUS "Appending -O2 flag for ${_source}")
endforeach()
```

5. 为了确保设置属性，再次循环并在打印每个源文件的`COMPILE_FLAGS`属性:

```cmake
message(STATUS "Querying sources properties using plain syntax:")
foreach(_source ${sources_with_lower_optimization})
  get_source_file_property(_flags ${_source} COMPILE_FLAGS)
  message(STATUS "Source ${_source} has the following extra COMPILE_FLAGS: ${_flags}")
endforeach()
```

6. 最后，添加`compute-areas`可执行目标，并将`geometry`库连接上去:

```cmake
add_executable(compute-areas compute-areas.cpp)
target_link_libraries(compute-areas geometry)
```

7. 验证在配置步骤中正确设置了标志:

```bash
$ mkdir -p build
$ cd build
$ cmake ..

...
-- Setting source properties using IN LISTS syntax:
-- Appending -O2 flag for geometry_circle.cpp
-- Appending -O2 flag for geometry_rhombus.cpp
-- Querying sources properties using plain syntax:
-- Source geometry_circle.cpp has the following extra COMPILE_FLAGS: -O2
-- Source geometry_rhombus.cpp has the following extra COMPILE_FLAGS: -O2
```

8. 最后，还使用`VERBOSE=1`检查构建步骤。将看到`-O2`标志添加在`-O3`标志之后，但是最后一个优化级别标志(在本例中是`-O2`)不同:

```bash
$ cmake --build . -- VERBOSE=1
```

### 工作原理

`foreach-endforeach`语法可用于在变量列表上，表示重复特定任务。本示例中，使用它来操作、设置和获取项目中特定文件的编译器标志。CMake代码片段中引入了另外两个新命令:

- `set_source_files_properties(file PROPERTIES property value)`，它将属性设置为给定文件的传递值。与目标非常相似，文件在CMake中也有属性，允许对构建系统进行非常细粒度的控制。源文件的可用属性列表可以在这里找到: https://cmake.org/cmake/help/v3.5/manual/cmake-properties.7.html#source-file-properties 。
- `get_source_file_property(VAR file property)`，检索给定文件所需属性的值，并将其存储在CMake`VAR`变量中。

**NOTE**:*CMake中，列表是用分号分隔的字符串组。列表可以由`list`或`set`命令创建。例如，`set(var a b c d e)`和`list(APPEND a b c d e)`都创建了列表`a;b;c;d;e`。*

**TIPS**:*为了对一组文件降低优化，将它们收集到一个单独的目标(库)中，并为这个目标显式地设置优化级别，而不是附加一个标志，这样可能会更简洁，不过在本示例中，我们的重点是`foreach-endforeach`。*

### 更多信息

`foreach()`的四种使用方式:

- `foreach(loop_var arg1 arg2 ...)`: 其中提供循环变量和显式项列表。当为`sources_with_lower_optimization`中的项打印编译器标志集时，使用此表单。注意，如果项目列表位于变量中，则必须显式展开它；也就是说，`${sources_with_lower_optimization}`必须作为参数传递。
- 通过指定一个范围，可以对整数进行循环，例如：`foreach(loop_var range total)`或`foreach(loop_var range start stop [step])`。
- 对列表值变量的循环，例如：`foreach(loop_var IN LISTS [list1[...]])` 。参数解释为列表，其内容就会自动展开。
- 对变量的循环，例如：`foreach(loop_var IN ITEMS [item1 [...]])`。参数的内容没有展开。





# CMake 完整使用教程 之三 检测环境



- 本章的主要内容有：

  - 检测操作系统
  - 处理与平台相关的源码
  - 处理与编译器相关的源码
  - 检测处理器体系结构
  - 检测处理器指令集
  - 为Eigen库使能向量化

  尽管CMake跨平台，但有时源代码并不是完全可移植(例如：当使用依赖于供应商的扩展时)，我们努力使源代码能够跨平台、操作系统和编译器。这个过程中会发现，有必要根据平台不同的方式配置和/或构建代码。这对于历史代码或交叉编译尤其重要，我们将在第13章中讨论这个主题。了解处理器指令集也有助于优化特定目标平台的性能。本章会介绍，检测环境的方法，并给出建议。

## 检测操作系统

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-02/recipe-01 中找到。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

CMake是一组跨平台工具。不过，了解操作系统(OS)上执行配置或构建步骤也很重要。从而与操作系统相关的CMake代码，会根据操作系统启用条件编译，或者在可用或必要时使用特定于编译器的扩展。本示例中，我们将通过一个不需要编译任何源代码的示例，演示如何使用CMake检测操作系统。为了简单起见，我们只考虑配置过程。

### 具体实施

我们将用一个非常简单的`CMakeLists.txt`进行演示:

1. 首先，定义CMake最低版本和项目名称。请注意，语言是`NONE`:

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(recipe-01 LANGUAGES NONE)
```

2. 然后，根据检测到的操作系统信息打印消息:

```cmake
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	message(STATUS "Configuring on/for Linux")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	message(STATUS "Configuring on/for macOS")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
	message(STATUS "Configuring on/for Windows")
elseif(CMAKE_SYSTEM_NAME STREQUAL "AIX")
	message(STATUS "Configuring on/for IBM AIX")
else()
	message(STATUS "Configuring on/for ${CMAKE_SYSTEM_NAME}")
endif()
```

测试之前，检查前面的代码块，并考虑相应系统上的具体行为。

3. 现在，测试配置项目:

```bash
$ mkdir -p build
$ cd build
$ cmake ..
```

4. 关于CMake输出，这里有一行很有趣——在Linux系统上(在其他系统上，输出会不同):

```bash
-- Configuring on/for Linux
```

### 工作原理

CMake为目标操作系统定义了`CMAKE_SYSTEM_NAME`，因此不需要使用定制命令、工具或脚本来查询此信息。然后，可以使用此变量的值实现特定于操作系统的条件和解决方案。在具有`uname`命令的系统上，将此变量设置为`uname -s`的输出。该变量在macOS上设置为“Darwin”。在Linux和Windows上，它分别计算为“Linux”和“Windows”。我们了解了如何在特定的操作系统上执行特定的CMake代码。当然，应该尽量减少这种定制化行为，以便简化迁移到新平台的过程。

**NOTE**:*为了最小化从一个平台转移到另一个平台时的成本，应该避免直接使用Shell命令，还应该避免显式的路径分隔符(Linux和macOS上的前斜杠和Windows上的后斜杠)。CMake代码中只使用前斜杠作为路径分隔符，CMake将自动将它们转换为所涉及的操作系统环境。*

## 处理与平台相关的源代码

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-02/recipe-02 中找到，包含一个C++示例。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

理想情况下，应该避免依赖于平台的源代码，但是有时我们没有选择，特别是当要求配置和编译不是自己编写的代码时。本示例中，将演示如何使用CMake根据操作系统编译源代码。

### 准备工作

修改`hello-world.cpp`示例代码，将第1章第1节的例子进行修改:

```cpp
#include <cstdlib>
#include <iostream>
#include <string>

std::string say_hello() {
#ifdef IS_WINDOWS
  return std::string("Hello from Windows!");
#elif IS_LINUX
  return std::string("Hello from Linux!");
#elif IS_MACOS
  return std::string("Hello from macOS!");
#else
  return std::string("Hello from an unknown system!");
#endif
}

int main() {
  std::cout << say_hello() << std::endl;
  return EXIT_SUCCESS;
}
```

### 具体实施

完成一个`CMakeLists.txt`实例，使我们能够基于目标操作系统有条件地编译源代码：

1. 首先，设置了CMake最低版本、项目名称和支持的语言:

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(recipe-02 LANGUAGES CXX)
```

2. 然后，定义可执行文件及其对应的源文件:

```cmake
add_executable(hello-world hello-world.cpp)
```

3. 通过定义以下目标编译定义，让预处理器知道系统名称:

```cmake
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_compile_definitions(hello-world PUBLIC "IS_LINUX")
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  target_compile_definitions(hello-world PUBLIC "IS_MACOS")
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  target_compile_definitions(hello-world PUBLIC "IS_WINDOWS")
endif()
```

继续之前，先检查前面的表达式，并考虑在不同系统上有哪些行为。

4. 现在，准备测试它，并配置项目:

```bash
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
$ ./hello-world

Hello from Linux!

```

Windows系统上，将看到来自Windows的Hello。其他操作系统将产生不同的输出。

### 工作原理

`hello-world.cpp`示例中，有趣的部分是基于预处理器定义`IS_WINDOWS`、`IS_LINUX`或`IS_MACOS`的条件编译:

```cpp
std::string say_hello() {
#ifdef IS_WINDOWS
  return std::string("Hello from Windows!");
#elif IS_LINUX
  return std::string("Hello from Linux!");
#elif IS_MACOS
  return std::string("Hello from macOS!");
#else
  return std::string("Hello from an unknown system!");
#endif
}
```

这些定义在CMakeLists.txt中配置时定义，通过使用`target_compile_definition`在预处理阶段使用。可以不重复`if-endif`语句，以更紧凑的表达式实现，我们将在下一个示例中演示这种重构方式。也可以把`if-endif`语句加入到一个`if-else-else-endif`语句中。这个阶段，可以使用`add_definitions(-DIS_LINUX)`来设置定义(当然，可以根据平台调整定义)，而不是使用`target_compile_definition`。使用`add_definitions`的缺点是，会修改编译整个项目的定义，而`target_compile_definitions`给我们机会，将定义限制于一个特定的目标，以及通过`PRIVATE|PUBLIC|INTERFACE`限定符，限制这些定义可见性。第1章的第8节，对这些限定符有详细的说明:

- **PRIVATE**，编译定义将只应用于给定的目标，而不应用于相关的其他目标。
- **INTERFACE**，对给定目标的编译定义将只应用于使用它的目标。
- **PUBLIC**，编译定义将应用于给定的目标和使用它的所有其他目标。

**NOTE**:*将项目中的源代码与平台相关性最小化，可使移植更加容易。*

##  处理与编译器相关的源代码

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-02/recipe-03 中找到，包含一个C++和Fortran示例。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

这个方法与前面的方法类似，我们将使用CMake来编译依赖于环境的条件源代码：本例将依赖于编译器。为了可移植性，我们尽量避免去编写新代码，但遇到有依赖的情况我们也要去解决，特别是当使用历史代码或处理编译器依赖工具，如[sanitizers](https://github.com/google/sanitizers)。从这一章和前一章的示例中，我们已经掌握了实现这一目标的所有方法。尽管如此，讨论与编译器相关的源代码的处理问题还是很有用的，这样我们将有机会从另一方面了解CMake。

### 准备工作

本示例中，我们将从`C++`中的一个示例开始，稍后我们将演示一个`Fortran`示例，并尝试重构和简化CMake代码。

看一下`hello-world.cpp`源代码:

```cpp
#include <cstdlib>
#include <iostream>
#include <string>

std::string say_hello() {
#ifdef IS_INTEL_CXX_COMPILER
  // only compiled when Intel compiler is selected
  // such compiler will not compile the other branches
  return std::string("Hello Intel compiler!");
#elif IS_GNU_CXX_COMPILER
  // only compiled when GNU compiler is selected
  // such compiler will not compile the other branches
  return std::string("Hello GNU compiler!");
#elif IS_PGI_CXX_COMPILER
  // etc.
  return std::string("Hello PGI compiler!");
#elif IS_XL_CXX_COMPILER
  return std::string("Hello XL compiler!");
#else
  return std::string("Hello unknown compiler - have we met before?");
#endif
}

int main() {
  std::cout << say_hello() << std::endl;
  std::cout << "compiler name is " COMPILER_NAME << std::endl;
  return EXIT_SUCCESS;
}
```

`Fortran`示例(`hello-world.F90`):

```fortran
program hello

  implicit none
#ifdef IS_Intel_FORTRAN_COMPILER
  print *, 'Hello Intel compiler!'
#elif IS_GNU_FORTRAN_COMPILER
  print *, 'Hello GNU compiler!'
#elif IS_PGI_FORTRAN_COMPILER
  print *, 'Hello PGI compiler!'
#elif IS_XL_FORTRAN_COMPILER
  print *, 'Hello XL compiler!'
#else
  print *, 'Hello unknown compiler - have we met before?'
#endif

end program
```

### 具体实施

我们将从`C++`的例子开始，然后再看`Fortran`的例子:

1. `CMakeLists.txt`文件中，定义了CMake最低版本、项目名称和支持的语言:

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(recipe-03 LANGUAGES CXX)
```

2. 然后，定义可执行目标及其对应的源文件:

```cmake
add_executable(hello-world hello-world.cpp)
```

3. 通过定义以下目标编译定义，让预处理器了解编译器的名称和供应商:

```cmake
target_compile_definitions(hello-world PUBLIC "COMPILER_NAME=\"${CMAKE_CXX_COMPILER_ID}\"")

if(CMAKE_CXX_COMPILER_ID MATCHES Intel)
  target_compile_definitions(hello-world PUBLIC "IS_INTEL_CXX_COMPILER")
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
  target_compile_definitions(hello-world PUBLIC "IS_GNU_CXX_COMPILER")
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES PGI)
  target_compile_definitions(hello-world PUBLIC "IS_PGI_CXX_COMPILER")
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES XL)
  target_compile_definitions(hello-world PUBLIC "IS_XL_CXX_COMPILER")
endif()
```

现在我们已经可以预测结果了:

```bash
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
$ ./hello-world

Hello GNU compiler!
```

使用不同的编译器，此示例代码将打印不同的问候语。

前一个示例的`CMakeLists.txt`文件中的`if`语句似乎是重复的，我们不喜欢重复的语句。能更简洁地表达吗？当然可以！为此，让我们再来看看`Fortran`示例。

`Fortran`例子的`CMakeLists.txt`文件中，我们需要做以下工作:

1. 需要使`Fortran`语言:

```cmake
project(recipe-03 LANGUAGE Fortran)
```

2. 然后，定义可执行文件及其对应的源文件。在本例中，使用大写`.F90`后缀:

```cmake
add_executable(hello-world hello-world.F90)
```

3. 我们通过定义下面的目标编译定义，让预处理器非常清楚地了解编译器:

```cmake
target_compile_definitions(hello-world
  PUBLIC "IS_${CMAKE_Fortran_COMPILER_ID}_FORTRAN_COMPILER"
  )
```

其余行为与`C++`示例相同。

### 工作原理

`CMakeLists.txt`会在配置时，进行预处理定义，并传递给预处理器。`Fortran`示例包含非常紧凑的表达式，我们使用`CMAKE_Fortran_COMPILER_ID`变量，通过`target_compile_definition`使用构造预处理器进行预处理定义。为了适应这种情况，我们必须将”Intel”从`IS_INTEL_CXX_COMPILER`更改为`IS_Intel_FORTRAN_COMPILER`。通过使用相应的`CMAKE_C_COMPILER_ID`和`CMAKE_CXX_COMPILER_ID`变量，我们可以在`C`或`C++`中实现相同的效果。但是，请注意，`CMAKE_<LANG>_COMPILER_ID`不能保证为所有编译器或语言都定义。

**NOTE**:*对于应该预处理的`Fortran`代码使用`.F90`后缀，对于不需要预处理的代码使用`.f90`后缀。*

## 检测处理器体系结构

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-02/recipe-04 中找到，包含一个C++示例。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

19世纪70年代，出现的64位整数运算和本世纪初出现的用于个人计算机的64位寻址，扩大了内存寻址范围，开发商投入了大量资源来移植为32位体系结构硬编码，以支持64位寻址。许多博客文章，如 https://www.viva64.com/en/a/0004/ ，致力于讨论将`C++`代码移植到64位平台中的典型问题和解决方案。虽然，避免显式硬编码的方式非常明智，但需要在使用CMake配置的代码中适应硬编码限制。本示例中，我们会来讨论检测主机处理器体系结构的选项。

### 准备工作

我们以下面的`arch-dependent.cpp`代码为例：

```c++
#include <cstdlib>
#include <iostream>
#include <string>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

std::string say_hello()
{
  std::string arch_info(TOSTRING(ARCHITECTURE));
  arch_info += std::string(" architecture. ");
#ifdef IS_32_BIT_ARCH
  return arch_info + std::string("Compiled on a 32 bit host processor.");
#elif IS_64_BIT_ARCH
  return arch_info + std::string("Compiled on a 64 bit host processor.");
#else
  return arch_info + std::string("Neither 32 nor 64 bit, puzzling ...");
#endif
}

int main()
{
  std::cout << say_hello() << std::endl;
  return EXIT_SUCCESS;
}
```

### 具体实施

`CMakeLists.txt`文件中，我们需要以下内容:

1. 首先，定义可执行文件及其源文件依赖关系:

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(recipe-04 LANGUAGES CXX)
add_executable(arch-dependent arch-dependent.cpp)
```

2. 检查空指针类型的大小。CMake的`CMAKE_SIZEOF_VOID_P`变量会告诉我们CPU是32位还是64位。我们通过状态消息让用户知道检测到的大小，并设置预处理器定义:

```cmake
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  target_compile_definitions(arch-dependent PUBLIC "IS_64_BIT_ARCH")
  message(STATUS "Target is 64 bits")
else()
  target_compile_definitions(arch-dependent PUBLIC "IS_32_BIT_ARCH")
  message(STATUS "Target is 32 bits")
endif()
```

3. 通过定义以下目标编译定义，让预处理器了解主机处理器架构，同时在配置过程中打印状态消息:

```cmake
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "i386")
	message(STATUS "i386 architecture detected")
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "i686")
	message(STATUS "i686 architecture detected")
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
	message(STATUS "x86_64 architecture detected")
else()
	message(STATUS "host processor architecture is unknown")
endif()
target_compile_definitions(arch-dependent
  PUBLIC "ARCHITECTURE=${CMAKE_HOST_SYSTEM_PROCESSOR}"
  )
```

4. 配置项目，并注意状态消息(打印出的信息可能会发生变化):

```bash
$ mkdir -p build
$ cd build
$ cmake ..

...
-- Target is 64 bits
-- x86_64 architecture detected
...
```

5. 最后，构建并执行代码(实际输出将取决于处理器架构):

```bash
$ cmake --build .
$ ./arch-dependent

x86_64 architecture. Compiled on a 64 bit host processor.
```

### 工作原理

CMake定义了`CMAKE_HOST_SYSTEM_PROCESSOR`变量，以包含当前运行的处理器的名称。可以设置为“i386”、“i686”、“x86_64”、“AMD64”等等，当然，这取决于当前的CPU。`CMAKE_SIZEOF_VOID_P`为void指针的大小。我们可以在CMake配置时进行查询，以便修改目标或目标编译定义。可以基于检测到的主机处理器体系结构，使用预处理器定义，确定需要编译的分支源代码。正如在前面的示例中所讨论的，编写新代码时应该避免这种依赖，但在处理遗留代码或交叉编译时，这种依赖是有用的，交叉编译会在第13章进行讨论。

**NOTE**:*使用`CMAKE_SIZEOF_VOID_P`是检查当前CPU是否具有32位或64位架构的唯一“真正”可移植的方法。*

### 更多信息

除了`CMAKE_HOST_SYSTEM_PROCESSOR`, CMake还定义了`CMAKE_SYSTEM_PROCESSOR`变量。前者包含当前运行的CPU在CMake的名称，而后者将包含当前正在为其构建的CPU的名称。这是一个细微的差别，在交叉编译时起着非常重要的作用。我们将在第13章，看到更多关于交叉编译的内容。另一种让CMake检测主机处理器体系结构，是使用`C`或`C++中`定义的符号，结合CMake的`try_run`函数，尝试构建执行的源代码(见第5.8节)分支的预处理符号。这将返回已定义错误码，这些错误可以在CMake端捕获(此策略的灵感来自 https://github.com/axr/cmake/blob/master/targetarch.cmake ):

```c++
#if defined(__i386) || defined(__i386__) || defined(_M_IX86)
	#error cmake_arch i386
#elif defined(__x86_64) || defined(__x86_64__) || defined(__amd64) || defined(_M_X64)
	#error cmake_arch x86_64
#endif
```

这种策略也是检测目标处理器体系结构的推荐策略，因为CMake似乎没有提供可移植的内在解决方案。另一种选择，将只使用CMake，完全不使用预处理器，代价是为每种情况设置不同的源文件，然后使用`target_source`命令将其设置为可执行目标`arch-dependent`依赖的源文件:

```cmake
add_executable(arch-dependent "")

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "i386")
	message(STATUS "i386 architecture detected")
	target_sources(arch-dependent
		PRIVATE
		arch-dependent-i386.cpp
	)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "i686")
	message(STATUS "i686 architecture detected")
	target_sources(arch-dependent
		PRIVATE
			arch-dependent-i686.cpp
	)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
	message(STATUS "x86_64 architecture detected")
	target_sources(arch-dependent
		PRIVATE
			arch-dependent-x86_64.cpp
	)
else()
	message(STATUS "host processor architecture is unknown")
endif()
```

这种方法，显然需要对现有项目进行更多的工作，因为源文件需要分离。此外，不同源文件之间的代码复制肯定也会成为问题。

## 检测处理器指令集

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-02/recipe-05 中找到，包含一个C++示例。该示例在CMake 3.10版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

本示例中，我们将讨论如何在CMake的帮助下检测主机处理器支持的指令集。这个功能是较新版本添加到CMake中的，需要CMake 3.10或更高版本。检测到的主机系统信息，可用于设置相应的编译器标志，或实现可选的源代码编译，或根据主机系统生成源代码。本示例中，我们的目标是检测主机系统信息，使用预处理器定义将其传递给`C++`源代码，并将信息打印到输出中。

### 准备工作

我们是`C++`源码(`processor-info.cpp`)如下所示：

```c++
#include "config.h"

#include <cstdlib>
#include <iostream>

int main()
{
  std::cout << "Number of logical cores: "
            << NUMBER_OF_LOGICAL_CORES << std::endl;
  std::cout << "Number of physical cores: "
            << NUMBER_OF_PHYSICAL_CORES << std::endl;
  std::cout << "Total virtual memory in megabytes: "
            << TOTAL_VIRTUAL_MEMORY << std::endl;
  std::cout << "Available virtual memory in megabytes: "
            << AVAILABLE_VIRTUAL_MEMORY << std::endl;
  std::cout << "Total physical memory in megabytes: "
            << TOTAL_PHYSICAL_MEMORY << std::endl;
  std::cout << "Available physical memory in megabytes: "
            << AVAILABLE_PHYSICAL_MEMORY << std::endl;
  std::cout << "Processor is 64Bit: "
            << IS_64BIT << std::endl;
  std::cout << "Processor has floating point unit: "
            << HAS_FPU << std::endl;
  std::cout << "Processor supports MMX instructions: "
            << HAS_MMX << std::endl;
  std::cout << "Processor supports Ext. MMX instructions: "
            << HAS_MMX_PLUS << std::endl;
  std::cout << "Processor supports SSE instructions: "
            << HAS_SSE << std::endl;
  std::cout << "Processor supports SSE2 instructions: "
            << HAS_SSE2 << std::endl;
  std::cout << "Processor supports SSE FP instructions: "
            << HAS_SSE_FP << std::endl;
  std::cout << "Processor supports SSE MMX instructions: "
            << HAS_SSE_MMX << std::endl;
  std::cout << "Processor supports 3DNow instructions: "
            << HAS_AMD_3DNOW << std::endl;
  std::cout << "Processor supports 3DNow+ instructions: "
            << HAS_AMD_3DNOW_PLUS << std::endl;
  std::cout << "IA64 processor emulating x86 : "
            << HAS_IA64 << std::endl;
  std::cout << "OS name: "
            << OS_NAME << std::endl;
  std::cout << "OS sub-type: "
            << OS_RELEASE << std::endl;
  std::cout << "OS build ID: "
            << OS_VERSION << std::endl;
  std::cout << "OS platform: "
            << OS_PLATFORM << std::endl;
  return EXIT_SUCCESS;
}
```

其包含`config.h`头文件，我们将使用`config.h.in`生成这个文件。`config.h.in`如下:

```ini
#pragma once

#define NUMBER_OF_LOGICAL_CORES @_NUMBER_OF_LOGICAL_CORES@
#define NUMBER_OF_PHYSICAL_CORES @_NUMBER_OF_PHYSICAL_CORES@
#define TOTAL_VIRTUAL_MEMORY @_TOTAL_VIRTUAL_MEMORY@
#define AVAILABLE_VIRTUAL_MEMORY @_AVAILABLE_VIRTUAL_MEMORY@
#define TOTAL_PHYSICAL_MEMORY @_TOTAL_PHYSICAL_MEMORY@
#define AVAILABLE_PHYSICAL_MEMORY @_AVAILABLE_PHYSICAL_MEMORY@
#define IS_64BIT @_IS_64BIT@
#define HAS_FPU @_HAS_FPU@
#define HAS_MMX @_HAS_MMX@
#define HAS_MMX_PLUS @_HAS_MMX_PLUS@
#define HAS_SSE @_HAS_SSE@
#define HAS_SSE2 @_HAS_SSE2@
#define HAS_SSE_FP @_HAS_SSE_FP@
#define HAS_SSE_MMX @_HAS_SSE_MMX@
#define HAS_AMD_3DNOW @_HAS_AMD_3DNOW@
#define HAS_AMD_3DNOW_PLUS @_HAS_AMD_3DNOW_PLUS@
#define HAS_IA64 @_HAS_IA64@
#define OS_NAME "@_OS_NAME@"
#define OS_RELEASE "@_OS_RELEASE@"
#define OS_VERSION "@_OS_VERSION@"
#define OS_PLATFORM "@_OS_PLATFORM@"
```

### 如何实施

我们将使用CMake为平台填充`config.h`中的定义，并将示例源文件编译为可执行文件:

1. 首先，我们定义了CMake最低版本、项目名称和项目语言:

```cmake
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(recipe-05 CXX)
```

2. 然后，定义目标可执行文件及其源文件，并包括目录:

```cmake
add_executable(processor-info "")

target_sources(processor-info
  PRIVATE
  	processor-info.cpp
  )

target_include_directories(processor-info
  PRIVATE
 	  ${PROJECT_BINARY_DIR}
  )
```

3. 继续查询主机系统的信息，获取一些关键字:

```cmake
foreach(key
  IN ITEMS
    NUMBER_OF_LOGICAL_CORES
    NUMBER_OF_PHYSICAL_CORES
    TOTAL_VIRTUAL_MEMORY
    AVAILABLE_VIRTUAL_MEMORY
    TOTAL_PHYSICAL_MEMORY
    AVAILABLE_PHYSICAL_MEMORY
    IS_64BIT
    HAS_FPU
    HAS_MMX
    HAS_MMX_PLUS
    HAS_SSE
    HAS_SSE2
    HAS_SSE_FP
    HAS_SSE_MMX
    HAS_AMD_3DNOW
    HAS_AMD_3DNOW_PLUS
    HAS_IA64
    OS_NAME
    OS_RELEASE
    OS_VERSION
    OS_PLATFORM
  )
  cmake_host_system_information(RESULT _${key} QUERY ${key})
endforeach()
```

4. 定义了相应的变量后，配置`config.h`:

```cmake
configure_file(config.h.in config.h @ONLY)
```

5. 现在准备好配置、构建和测试项目:

```cmake
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
$ ./processor-info

Number of logical cores: 4
Number of physical cores: 2
Total virtual memory in megabytes: 15258
Available virtual memory in megabytes: 14678
Total physical memory in megabytes: 7858
Available physical memory in megabytes: 4072
Processor is 64Bit: 1
Processor has floating point unit: 1
Processor supports MMX instructions: 1
Processor supports Ext. MMX instructions: 0
Processor supports SSE instructions: 1
Processor supports SSE2 instructions: 1
Processor supports SSE FP instructions: 0
Processor supports SSE MMX instructions: 0
Processor supports 3DNow instructions: 0
Processor supports 3DNow+ instructions: 0
IA64 processor emulating x86 : 0
OS name: Linux
OS sub-type: 4.16.7-1-ARCH
OS build ID: #1 SMP PREEMPT Wed May 2 21:12:36 UTC 2018
OS platform: x86_64
```

6. 输出会随着处理器的不同而变化。

### 工作原理

`CMakeLists.txt`中的`foreach`循环会查询多个键值，并定义相应的变量。此示例的核心函数是`cmake_host_system_information`，它查询运行CMake的主机系统的系统信息。本例中，我们对每个键使用了一个函数调用。然后，使用这些变量来配置`config.h.in`中的占位符，输入并生成`config.h`。此配置使用`configure_file`命令完成。最后，`config.h`包含在`processor-info.cpp`中。编译后，它将把值打印到屏幕上。我们将在第5章(配置时和构建时操作)和第6章(生成源代码)中重新讨论这种方法。

### 更多信息

对于更细粒度的处理器指令集检测，请考虑以下模块: https://github.com/VcDevel/Vc/blob/master/cmake/OptimizeForArchitecture.cmake 。有时候，构建代码的主机可能与运行代码的主机不一样。在计算集群中，登录节点的体系结构可能与计算节点上的体系结构不同。解决此问题的一种方法是，将配置和编译作为计算步骤，提交并部署到相应计算节点上。

##  为Eigen库使能向量化

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-02/recipe-06 中找到，包含一个C++示例。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

处理器的向量功能，可以提高代码的性能。对于某些类型的运算来说尤为甚之，例如：线性代数。本示例将展示如何使能矢量化，以便使用线性代数的Eigen C++库加速可执行文件。

### 准备工作

我们用Eigen C++模板库，用来进行线性代数计算，并展示如何设置编译器标志来启用向量化。这个示例的源代码`linear-algebra.cpp`文件:

```c++
#include <chrono>
#include <iostream>

#include <Eigen/Dense>

EIGEN_DONT_INLINE
double simple_function(Eigen::VectorXd &va, Eigen::VectorXd &vb)
{
  // this simple function computes the dot product of two vectors
  // of course it could be expressed more compactly
  double d = va.dot(vb);
  return d;
}

int main()
{
  int len = 1000000;
  int num_repetitions = 100;
  
  // generate two random vectors
  Eigen::VectorXd va = Eigen::VectorXd::Random(len);
  Eigen::VectorXd vb = Eigen::VectorXd::Random(len);
  
  double result;
  auto start = std::chrono::system_clock::now();
  for (auto i = 0; i < num_repetitions; i++)
  {
    result = simple_function(va, vb);
  }
  auto end = std::chrono::system_clock::now();
  auto elapsed_seconds = end - start;
  
  std::cout << "result: " << result << std::endl;
  std::cout << "elapsed seconds: " << elapsed_seconds.count() << std::endl;
}
```

我们期望向量化可以加快`simple_function`中的点积操作。

### 如何实施

根据Eigen库的文档，设置适当的编译器标志就足以生成向量化的代码。让我们看看`CMakeLists.txt`:

1. 声明一个`C++11`项目:

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-06 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

2. 使用Eigen库，我们需要在系统上找到它的头文件:

```cmake
find_package(Eigen3 3.3 REQUIRED CONFIG)
```

3. `CheckCXXCompilerFlag.cmake`标准模块文件:

```cmake
include(CheckCXXCompilerFlag)
```

4. 检查`-march=native`编译器标志是否工作:

```cmake
check_cxx_compiler_flag("-march=native" _march_native_works)
```

5. 另一个选项`-xHost`编译器标志也开启:

```cmake
check_cxx_compiler_flag("-xHost" _xhost_works)
```

6. 设置了一个空变量`_CXX_FLAGS`，来保存刚才检查的两个编译器中找到的编译器标志。如果看到`_march_native_works`，我们将`_CXX_FLAGS`设置为`-march=native`。如果看到`_xhost_works`，我们将`_CXX_FLAGS`设置为`-xHost`。如果它们都不起作用，`_CXX_FLAGS`将为空，并禁用矢量化:



```cmake
set(_CXX_FLAGS)
if(_march_native_works)
	message(STATUS "Using processor's vector instructions (-march=native compiler flag set)")
	set(_CXX_FLAGS "-march=native")
elseif(_xhost_works)
	message(STATUS "Using processor's vector instructions (-xHost compiler flag set)")
	set(_CXX_FLAGS "-xHost")
else()
	message(STATUS "No suitable compiler flag found for vectorization")
endif()
```

7. 为了便于比较，我们还为未优化的版本定义了一个可执行目标，不使用优化标志:

```cmake
add_executable(linear-algebra-unoptimized linear-algebra.cpp)

target_link_libraries(linear-algebra-unoptimized
  PRIVATE
  	Eigen3::Eigen
  )
```

8. 此外，我们定义了一个优化版本:

```cmake
add_executable(linear-algebra linear-algebra.cpp)

target_compile_options(linear-algebra
  PRIVATE
  	${_CXX_FLAGS}
  )

target_link_libraries(linear-algebra
  PRIVATE
  	Eigen3::Eigen
  )
```

9. 让我们比较一下这两个可执行文件——首先我们配置(在本例中，`-march=native_works`):

```cmake
$ mkdir -p build
$ cd build
$ cmake ..

...
-- Performing Test _march_native_works
-- Performing Test _march_native_works - Success
-- Performing Test _xhost_works
-- Performing Test _xhost_works - Failed
-- Using processor's vector instructions (-march=native compiler flag set)
...
```

10. 最后，让我们编译可执行文件，并比较运行时间:

```bash
$ cmake --build .
$ ./linear-algebra-unoptimized

result: -261.505
elapsed seconds: 1.97964

$ ./linear-algebra

result: -261.505
elapsed seconds: 1.05048
```

### 工作原理

大多数处理器提供向量指令集，代码可以利用这些特性，获得更高的性能。由于线性代数运算可以从Eigen库中获得很好的加速，所以在使用Eigen库时，就要考虑向量化。我们所要做的就是，指示编译器为我们检查处理器，并为当前体系结构生成本机指令。不同的编译器供应商会使用不同的标志来实现这一点：GNU编译器使用`-march=native`标志来实现这一点，而Intel编译器使用`-xHost`标志。使用`CheckCXXCompilerFlag.cmake`模块提供的`check_cxx_compiler_flag`函数进行编译器标志的检查:

```cmake
check_cxx_compiler_flag("-march=native" _march_native_works)
```

这个函数接受两个参数:

- 第一个是要检查的编译器标志。
- 第二个是用来存储检查结果(true或false)的变量。如果检查为真，我们将工作标志添加到`_CXX_FLAGS`变量中，该变量将用于为可执行目标设置编译器标志。

### 更多信息

本示例可与前一示例相结合，可以使用`cmake_host_system_information`查询处理器功能。



# CMake 完整使用教程 之四 检测外部库和程序



本章中主要内容有:

- 检测Python解释器
- 检测Python库
- 检测Python模块和包
- 检测BLAS和LAPACK数学库
- 检测OpenMP并行环境
- 检测MPI并行环境
- 检测Eigen库
- 检测Boost库
- 检测外部库:Ⅰ. 使用pkg-config
- 检测外部库:Ⅱ. 书写find模块

我们的项目常常会依赖于其他项目和库。本章将演示，如何检测外部库、框架和项目，以及如何链接到这些库。CMake有一组预打包模块，用于检测常用库和程序，例如：Python和Boost。可以使用`cmake --help-module-list`获得现有模块的列表。但是，不是所有的库和程序都包含在其中，有时必须自己编写检测脚本。本章将讨论相应的工具，了解CMake的`find`族命令:

- **find_file**：在相应路径下查找命名文件
- **find_library**：查找一个库文件
- **find_package**：从外部项目查找和加载设置
- **find_path**：查找包含指定文件的目录
- **find_program**：找到一个可执行程序

**NOTE**:*可以使用`--help-command`命令行显示CMake内置命令的打印文档。*

## 检测Python解释器

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-03/recipe-01 中找到。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

Python是一种非常流行的语言。许多项目用Python编写的工具，从而将主程序和库打包在一起，或者在配置或构建过程中使用Python脚本。这种情况下，确保运行时对Python解释器的依赖也需要得到满足。本示例将展示如何检测和使用Python解释器。

我们将介绍`find_package`命令，这个命令将贯穿本章。

### 具体实施

我们将逐步建立`CMakeLists.txt`文件:

1. 首先，定义CMake最低版本和项目名称。注意，这里不需要任何语言支持:

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(recipe-01 LANGUAGES NONE)
```

2. 然后，使用`find_package`命令找到Python解释器:

```cmake
find_package(PythonInterp REQUIRED)
```

3. 然后，执行Python命令并捕获它的输出和返回值:

```cmake
execute_process(
  COMMAND
  	${PYTHON_EXECUTABLE} "-c" "print('Hello, world!')"
  RESULT_VARIABLE _status
  OUTPUT_VARIABLE _hello_world
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )
```

4. 最后，打印Python命令的返回值和输出:

```cmake
message(STATUS "RESULT_VARIABLE is: ${_status}")
message(STATUS "OUTPUT_VARIABLE is: ${_hello_world}")
```

5. 配置项目:

```cmake
$ mkdir -p build
$ cd build
$ cmake ..

-- Found PythonInterp: /usr/bin/python (found version "3.6.5")
-- RESULT_VARIABLE is: 0
-- OUTPUT_VARIABLE is: Hello, world!
-- Configuring done
-- Generating done
-- Build files have been written to: /home/user/cmake-cookbook/chapter-03/recipe-01/example/build
```

### 工作原理

`find_package`是用于发现和设置包的CMake模块的命令。这些模块包含CMake命令，用于标识系统标准位置中的包。CMake模块文件称为`Find<name>.cmake`，当调用`find_package(<name>)`时，模块中的命令将会运行。

除了在系统上实际查找包模块之外，查找模块还会设置了一些有用的变量，反映实际找到了什么，也可以在自己的`CMakeLists.txt`中使用这些变量。对于Python解释器，相关模块为`FindPythonInterp.cmake`附带的设置了一些CMake变量:

- **PYTHONINTERP_FOUND**：是否找到解释器
- **PYTHON_EXECUTABLE**：Python解释器到可执行文件的路径
- **PYTHON_VERSION_STRING**：Python解释器的完整版本信息
- **PYTHON_VERSION_MAJOR**：Python解释器的主要版本号
- **PYTHON_VERSION_MINOR** ：Python解释器的次要版本号
- **PYTHON_VERSION_PATCH**：Python解释器的补丁版本号

可以强制CMake，查找特定版本的包。例如，要求Python解释器的版本大于或等于2.7：`find_package(PythonInterp 2.7)`

可以强制满足依赖关系:

```cmake
find_package(PythonInterp REQUIRED)
```

如果在查找位置中没有找到适合Python解释器的可执行文件，CMake将中止配置。

**TIPS**:*CMake有很多查找软件包的模块。我们建议在CMake在线文档中查询`Find<package>.cmake`模块，并在使用它们之前详细阅读它们的文档。`find_package`命令的文档可以参考 https://cmake.org/cmake/help/v3.5/command/find_ackage.html 。在线文档的一个很好的替代方法是浏览 https://github.com/Kitware/CMake/tree/master/Modules 中的CMake模块源代码——它们记录了模块使用的变量，以及模块可以在`CMakeLists.txt`中使用的变量。*

### 更多信息

软件包没有安装在标准位置时，CMake无法正确定位它们。用户可以使用CLI的`-D`参数传递相应的选项，告诉CMake查看特定的位置。Python解释器可以使用以下配置:

```bash
$ cmake -D PYTHON_EXECUTABLE=/custom/location/python ..
```

这将指定非标准`/custom/location/python`安装目录中的Python可执行文件。

**NOTE**:*每个包都是不同的，`Find<package>.cmake`模块试图提供统一的检测接口。当CMake无法找到模块包时，我们建议您阅读相应检测模块的文档，以了解如何正确地使用CMake模块。可以在终端中直接浏览文档，本例中可使用`cmake --help-module FindPythonInterp`查看。*

除了检测包之外，我们还想提到一个便于打印变量的helper模块。本示例中，我们使用了以下方法:

```cmake
message(STATUS "RESULT_VARIABLE is: ${_status}")
message(STATUS "OUTPUT_VARIABLE is: ${_hello_world}")
```

使用以下工具进行调试:

```cmake
include(CMakePrintHelpers)
cmake_print_variables(_status _hello_world)
```

将产生以下输出:

```bash
-- _status="0" ; _hello_world="Hello, world!"
```

有关打印属性和变量的更多信息，请参考 https://cmake.org/cmake/help/v3.5/module/CMakePrintHelpers.html 。

##  检测Python库

**NOTE**:*此示例代码可以在 https://github.com/devcafe/cmake-cookbook/tree/v1.0/chapter-03/recipe-02 中找到，有一个C示例。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

可以使用Python工具来分析和操作程序的输出。然而，还有更强大的方法可以将解释语言(如Python)与编译语言(如C或C++)组合在一起使用。一种是扩展Python，通过编译成共享库的C或C++模块在这些类型上提供新类型和新功能，这是第9章的主题。另一种是将Python解释器嵌入到C或C++程序中。两种方法都需要下列条件:

- Python解释器的工作版本
- Python头文件Python.h的可用性
- Python运行时库libpython

三个组件所使用的Python版本必须相同。我们已经演示了如何找到Python解释器；本示例中，我们将展示另外两种方式。

### 准备工作

我们将一个简单的Python代码，嵌入到C程序中，可以在Python文档页面上找到。源文件称为`hello-embedded-python.c`:

```cmake
#include <Python.h>

int main(int argc, char *argv[]) {
  Py_SetProgramName(argv[0]); /* optional but recommended */
  Py_Initialize();
  PyRun_SimpleString("from time import time,ctime\n"
                     "print 'Today is',ctime(time())\n");
  Py_Finalize();
  return 0;
}
```

此代码将在程序中初始化Python解释器的实例，并使用Python的`time`模块，打印日期。

**NOTE**:*嵌入代码可以在Python文档页面的 https://docs.python.org/2/extending/embedding.html 和 https://docs.python.org/3/extending/embedding.html 中找到。*

### 具体实施

以下是`CMakeLists.txt`中的步骤:

1. 包含CMake最低版本、项目名称和所需语言:

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(recipe-02 LANGUAGES C)
```

2. 制使用C99标准，这不严格要求与Python链接，但有时你可能需要对Python进行连接:

```cmake
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_C_STANDARD_REQUIRED ON)
```

3. 找到Python解释器。这是一个`REQUIRED`依赖:

```cmake
find_package(PythonInterp REQUIRED)
```

4. 找到Python头文件和库的模块，称为`FindPythonLibs.cmake`:

```cmake
find_package(PythonLibs ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} EXACT REQUIRED)
```

5. 使用`hello-embedded-python.c`源文件，添加一个可执行目标:

```cmake
add_executable(hello-embedded-python hello-embedded-python.c)
```

6. 可执行文件包含`Python.h`头文件。因此，这个目标的`include`目录必须包含Python的`include`目录，可以通过`PYTHON_INCLUDE_DIRS`变量进行指定:

```cmake
target_include_directories(hello-embedded-python
  PRIVATE
  	${PYTHON_INCLUDE_DIRS}
	)
```

7. 最后，将可执行文件链接到Python库，通过`PYTHON_LIBRARIES`变量访问:

```cmake
target_link_libraries(hello-embedded-python
  PRIVATE
  	${PYTHON_LIBRARIES}
	)
```

8. 现在，进行构建:

```cmake
$ mkdir -p build
$ cd build
$ cmake ..

...
-- Found PythonInterp: /usr/bin/python (found version "3.6.5")
-- Found PythonLibs: /usr/lib/libpython3.6m.so (found suitable exact version "3.6.5")
```

9. 最后，执行构建，并运行可执行文件:

```bash
$ cmake --build .
$ ./hello-embedded-python

Today is Thu Jun 7 22:26:02 2018
```

### 工作原理

`FindPythonLibs.cmake`模块将查找Python头文件和库的标准位置。由于，我们的项目需要这些依赖项，如果没有找到这些依赖项，将停止配置，并报出错误。

注意，我们显式地要求CMake检测安装的Python可执行文件。这是为了确保可执行文件、头文件和库都有一个匹配的版本。这对于不同版本，可能在运行时导致崩溃。我们通过`FindPythonInterp.cmake`中定义的`PYTHON_VERSION_MAJOR`和`PYTHON_VERSION_MINOR`来实现:

```cmake
find_package(PythonInterp REQUIRED)
find_package(PythonLibs ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} EXACT REQUIRED)
```

使用`EXACT`关键字，限制CMake检测特定的版本，在本例中是匹配的相应Python版本的包括文件和库。我们可以使用`PYTHON_VERSION_STRING`变量，进行更接近的匹配:

```cmake
find_package(PythonInterp REQUIRED)
find_package(PythonLibs ${PYTHON_VERSION_STRING} EXACT REQUIRED)
```

### 更多信息

当Python不在标准安装目录中，我们如何确定Python头文件和库的位置是正确的？对于Python解释器，可以通过CLI的`-D`选项传递`PYTHON_LIBRARY`和`PYTHON_INCLUDE_DIR`选项来强制CMake查找特定的目录。这些选项指定了以下内容:

- **PYTHON_LIBRARY**：指向Python库的路径
- **PYTHON_INCLUDE_DIR**：Python.h所在的路径

这样，就能获得所需的Python版本。

**TIPS**:*有时需要将`-D PYTHON_EXECUTABLE`、`-D PYTHON_LIBRARY`和`-D PYTHON_INCLUDE_DIR`传递给CMake CLI，以便找到及定位相应的版本的组件。*

要将Python解释器及其开发组件匹配为完全相同的版本可能非常困难，对于那些将它们安装在非标准位置或系统上安装了多个版本的情况尤其如此。CMake 3.12版本中增加了新的Python检测模块，旨在解决这个棘手的问题。我们`CMakeLists.txt`的检测部分也将简化为:

```cmake
find_package(Python COMPONENTS Interpreter Development REQUIRED)
```

我们建议您阅读新模块的文档，地址是: https://cmake.org/cmake/help/v3.12/module/FindPython.html

## 检测Python模块和包

**NOTE**:*此示例代码可以在 https://github.com/devcafe/cmake-cookbook/tree/v1.0/chapter-03/recipe-03 中找到，包含一个C++示例。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

前面的示例中，我们演示了如何检测Python解释器，以及如何编译一个简单的C程序(嵌入Python解释器)。通常，代码将依赖于特定的Python模块，无论是Python工具、嵌入Python的程序，还是扩展Python的库。例如，科学界非常流行使用NumPy处理矩阵问题。依赖于Python模块或包的项目中，确定满足对这些Python模块的依赖非常重要。本示例将展示如何探测用户的环境，以找到特定的Python模块和包。

### 准备工作

我们将尝试在C++程序中嵌入一个稍微复杂一点的例子。这个示例再次引用[Python在线文档](https://docs.python.org/3.5/extending/embedding.html#pureembedded)，并展示了如何通过调用编译后的C++可执行文件，来执行用户定义的Python模块中的函数。

Python 3示例代码(`Py3-pure-embedding.cpp`)包含以下源代码(请参见https://docs.python.org/2/extending/embedding.html#pure-embedded 与Python 2代码等效):

```c++
#include <Python.h>
int main(int argc, char* argv[]) {
  PyObject* pName, * pModule, * pDict, * pFunc;
  PyObject* pArgs, * pValue;
  int i;
  if (argc < 3) {
    fprintf(stderr, "Usage: pure-embedding pythonfile funcname [args]\n");
    return 1;
  }
  Py_Initialize();
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append(\".\")");
  pName = PyUnicode_DecodeFSDefault(argv[1]);
  /* Error checking of pName left out */
  pModule = PyImport_Import(pName);
  Py_DECREF(pName);
  if (pModule != NULL) {
    pFunc = PyObject_GetAttrString(pModule, argv[2]);
    /* pFunc is a new reference */
    if (pFunc && PyCallable_Check(pFunc)) {
      pArgs = PyTuple_New(argc - 3);
      for (i = 0; i < argc - 3; ++i) {
        pValue = PyLong_FromLong(atoi(argv[i + 3]));
        if (!pValue) {
          Py_DECREF(pArgs);
          Py_DECREF(pModule);
          fprintf(stderr, "Cannot convert argument\n");
          return 1;
        }
        /* pValue reference stolen here: */
        PyTuple_SetItem(pArgs, i, pValue);
      }
      pValue = PyObject_CallObject(pFunc, pArgs);
      Py_DECREF(pArgs);
      if (pValue != NULL) {
        printf("Result of call: %ld\n", PyLong_AsLong(pValue));
        Py_DECREF(pValue);
      }
      else {
        Py_DECREF(pFunc);
        Py_DECREF(pModule);
        PyErr_Print();
        fprintf(stderr, "Call failed\n");
        return 1;
      }
    }
    else {
      if (PyErr_Occurred())
        PyErr_Print();
      fprintf(stderr, "Cannot find function \"%s\"\n", argv[2]);
    }
    Py_XDECREF(pFunc);
    Py_DECREF(pModule);
  }
  else {
    PyErr_Print();
    fprintf(stderr, "Failed to load \"%s\"\n", argv[1]);
    return 1;
  }
  Py_Finalize();
  return 0;
}
```



我们希望嵌入的Python代码(`use_numpy.py`)使用NumPy设置一个矩阵，所有矩阵元素都为1.0:

```python
import numpy as np
def print_ones(rows, cols):
  A = np.ones(shape=(rows, cols), dtype=float)
  print(A)
  
  # we return the number of elements to verify
  # that the C++ code is able to receive return values
  num_elements = rows*cols
  return(num_elements)
```

### 具体实施

下面的代码中，我们能够使用CMake检查NumPy是否可用。我们需要确保Python解释器、头文件和库在系统上是可用的。然后，将再来确认NumPy的可用性：

1. 首先，我们定义了最低CMake版本、项目名称、语言和C++标准:

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(recipe-03 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

2. 查找解释器、头文件和库的方法与前面的方法完全相同:

```cmake
find_package(PythonInterp REQUIRED)
find_package(PythonLibs ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} EXACT REQUIRED)
```

3. 正确打包的Python模块，指定安装位置和版本。可以在`CMakeLists.txt`中执行Python脚本进行探测:

```cmake
execute_process(
  COMMAND
  	${PYTHON_EXECUTABLE} "-c" "import re, numpy; print(re.compile('/__init__.py.*').sub('',numpy.__file__))"
  RESULT_VARIABLE _numpy_status
  OUTPUT_VARIABLE _numpy_location
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )
```

4. 如果找到NumPy，则`_numpy_status`变量为整数，否则为错误的字符串，而`_numpy_location`将包含NumPy模块的路径。如果找到NumPy，则将它的位置保存到一个名为`NumPy`的新变量中。注意，新变量被缓存，这意味着CMake创建了一个持久性变量，用户稍后可以修改该变量:

```cmake
if(NOT _numpy_status)
	set(NumPy ${_numpy_location} CACHE STRING "Location of NumPy")
endif()
```

5. 下一步是检查模块的版本。同样，我们在`CMakeLists.txt`中施加了一些Python魔法，将版本保存到`_numpy_version`变量中:

```cmake
execute_process(
  COMMAND
  	${PYTHON_EXECUTABLE} "-c" "import numpy; print(numpy.__version__)"
  OUTPUT_VARIABLE _numpy_version
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )
```

6. 最后，`FindPackageHandleStandardArgs`的CMake包以正确的格式设置`NumPy_FOUND`变量和输出信息:

```cmake
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NumPy
  FOUND_VAR NumPy_FOUND
  REQUIRED_VARS NumPy
  VERSION_VAR _numpy_version
  )
```

7. 一旦正确的找到所有依赖项，我们就可以编译可执行文件，并将其链接到Python库:

```cmake
dd_executable(pure-embedding "")

target_sources(pure-embedding
  PRIVATE
  	Py${PYTHON_VERSION_MAJOR}-pure-embedding.cpp
  )
  
target_include_directories(pure-embedding
  PRIVATE
  	${PYTHON_INCLUDE_DIRS}
  )
  
target_link_libraries(pure-embedding
  PRIVATE
  	${PYTHON_LIBRARIES}
  )
```

8. 我们还必须保证`use_numpy.py`在`build`目录中可用:

```cmake
add_custom_command(
  OUTPUT
  	${CMAKE_CURRENT_BINARY_DIR}/use_numpy.py
  COMMAND
  	${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/use_numpy.py
 	 ${CMAKE_CURRENT_BINARY_DIR}/use_numpy.py
  DEPENDS
  	${CMAKE_CURRENT_SOURCE_DIR}/use_numpy.py
  )
  
# make sure building pure-embedding triggers the above custom command
target_sources(pure-embedding
  PRIVATE
  	${CMAKE_CURRENT_BINARY_DIR}/use_numpy.py
  )
```

9. 现在，我们可以测试嵌入的代码:

```bash
$ mkdir -p build
$ cd build
$ cmake ..

-- ...
-- Found PythonInterp: /usr/bin/python (found version "3.6.5")
-- Found PythonLibs: /usr/lib/libpython3.6m.so (found suitable exact version "3.6.5")
-- Found NumPy: /usr/lib/python3.6/site-packages/numpy (found version "1.14.3")

$ cmake --build .
$ ./pure-embedding use_numpy print_ones 2 3

[[1. 1. 1.]
[1. 1. 1.]]
Result of call: 6
```

### 工作原理

例子中有三个新的CMake命令，需要`include(FindPackageHandleStandardArgs)`：

- `execute_process`
- `add_custom_command`
- `find_package_handle_standard_args`

`execute_process`将作为通过子进程执行一个或多个命令。最后，子进程返回值将保存到变量作为参数，传递给`RESULT_VARIABLE`，而管道标准输出和标准错误的内容将被保存到变量作为参数传递给`OUTPUT_VARIABLE`和`ERROR_VARIABLE`。`execute_process`可以执行任何操作，并使用它们的结果来推断系统配置。本例中，用它来确保NumPy可用，然后获得模块版本。

`find_package_handle_standard_args`提供了，用于处理与查找相关程序和库的标准工具。引用此命令时，可以正确的处理与版本相关的选项(`REQUIRED`和`EXACT`)，而无需更多的CMake代码。稍后将介绍`QUIET`和`COMPONENTS`选项。本示例中，使用了以下方法:

```cmake
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NumPy
  FOUND_VAR NumPy_FOUND
  REQUIRED_VARS NumPy
  VERSION_VAR _numpy_version
  )
```

所有必需的变量都设置为有效的文件路径(NumPy)后，发送到模块(`NumPy_FOUND`)。它还将版本保存在可传递的版本变量(`_numpy_version`)中并打印:

```bash
-- Found NumPy: /usr/lib/python3.6/site-packages/numpy (found version "1.14.3")
```

目前的示例中，没有进一步使用这些变量。如果返回`NumPy_FOUND`为`FALSE`，则停止配置。

最后，将`use_numpy.py`复制到`build`目录，对代码进行注释:

```cmake
add_custom_command(
  OUTPUT
  	${CMAKE_CURRENT_BINARY_DIR}/use_numpy.py
  COMMAND
  	${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/use_numpy.py
  	${CMAKE_CURRENT_BINARY_DIR}/use_numpy.py
  DEPENDS
  	${CMAKE_CURRENT_SOURCE_DIR}/use_numpy.py
  )
	
target_sources(pure-embedding
  PRIVATE
  	${CMAKE_CURRENT_BINARY_DIR}/use_numpy.py
  )
```

我们也可以使用`file(COPY…)`命令来实现复制。这里，我们选择使用`add_custom_command`，来确保文件在每次更改时都会被复制，而不仅仅是第一次运行配置时。我们将在第5章更详细地讨论`add_custom_command`。还要注意`target_sources`命令，它将依赖项添加到`${CMAKE_CURRENT_BINARY_DIR}/use_numpy.py`；这样做是为了确保构建目标，能够触发之前的命令。

## 检测BLAS和LAPACK数学库

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-03/recipe-04 中找到，有一个C++示例。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

许多数据算法严重依赖于矩阵和向量运算。例如：矩阵-向量和矩阵-矩阵乘法，求线性方程组的解，特征值和特征向量的计算或奇异值分解。这些操作在代码库中非常普遍，因为操作的数据量比较大，因此高效的实现有绝对的必要。幸运的是，有专家库可用：基本线性代数子程序(BLAS)和线性代数包(LAPACK)，为许多线性代数操作提供了标准API。供应商有不同的实现，但都共享API。虽然，用于数学库底层实现，实际所用的编程语言会随着时间而变化(Fortran、C、Assembly)，但是也都是Fortran调用接口。考虑到调用街扩，本示例中的任务要链接到这些库，并展示如何用不同语言编写的库。

### 准备工作

为了展示数学库的检测和连接，我们编译一个C++程序，将矩阵的维数作为命令行输入，生成一个随机的方阵**A**，一个随机向量**b**，并计算线性系统方程: **Ax = b**。另外，将对向量**b**的进行随机缩放。这里，需要使用的子程序是BLAS中的DSCAL和LAPACK中的DGESV来求线性方程组的解。示例C++代码的清单( `linear-algebra.cpp`)：

```c++
#include "CxxBLAS.hpp"
#include "CxxLAPACK.hpp"

#include <iostream>
#include <random>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cout << "Usage: ./linear-algebra dim" << std::endl;
    return EXIT_FAILURE;
  }
  
  // Generate a uniform distribution of real number between -1.0 and 1.0
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  
  // Allocate matrices and right-hand side vector
  int dim = std::atoi(argv[1]);
  std::vector<double> A(dim * dim);
  std::vector<double> b(dim);
  std::vector<int> ipiv(dim);
  // Fill matrix and RHS with random numbers between -1.0 and 1.0
  for (int r = 0; r < dim; r++) {
    for (int c = 0; c < dim; c++) {
      A[r + c * dim] = dist(mt);
    }
    b[r] = dist(mt);
  }
  
  // Scale RHS vector by a random number between -1.0 and 1.0
  C_DSCAL(dim, dist(mt), b.data(), 1);
  std::cout << "C_DSCAL done" << std::endl;
  
  // Save matrix and RHS
  std::vector<double> A1(A);
  std::vector<double> b1(b);
  int info;
  info = C_DGESV(dim, 1, A.data(), dim, ipiv.data(), b.data(), dim);
  std::cout << "C_DGESV done" << std::endl;
  std::cout << "info is " << info << std::endl;
  
  double eps = 0.0;
  for (int i = 0; i < dim; ++i) {
    double sum = 0.0;
    for (int j = 0; j < dim; ++j)
      sum += A1[i + j * dim] * b[j];
    eps += std::abs(b1[i] - sum);
  }
  std::cout << "check is " << eps << std::endl;
  
  return 0;
}
```

使用C++11的随机库来生成-1.0到1.0之间的随机分布。`C_DSCAL`和`C_DGESV`分别是到BLAS和LAPACK库的接口。为了避免名称混淆，将在下面来进一步讨论CMake模块：

文件`CxxBLAS.hpp`用`extern "C"`封装链接BLAS:

```c++
#pragma once
#include "fc_mangle.h"
#include <cstddef>
#ifdef __cplusplus
extern "C" {
#endif
extern void DSCAL(int *n, double *alpha, double *vec, int *inc);
#ifdef __cplusplus
}
#endif
void C_DSCAL(size_t length, double alpha, double *vec, int inc);

```

对应的实现文件`CxxBLAS.cpp`:

```c++
#include "CxxBLAS.hpp"

#include <climits>

// see http://www.netlib.no/netlib/blas/dscal.f
void C_DSCAL(size_t length, double alpha, double *vec, int inc) {
  int big_blocks = (int)(length / INT_MAX);
  int small_size = (int)(length % INT_MAX);
  for (int block = 0; block <= big_blocks; block++) {
    double *vec_s = &vec[block * inc * (size_t)INT_MAX];
    signed int length_s = (block == big_blocks) ? small_size : INT_MAX;
    ::DSCAL(&length_s, &alpha, vec_s, &inc);
  }
}
```

`CxxLAPACK.hpp`和`CxxLAPACK.cpp`为LAPACK调用执行相应的转换。

### 具体实施

对应的`CMakeLists.txt`包含以下构建块:

1. 我们定义了CMake最低版本，项目名称和支持的语言:

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(recipe-04 LANGUAGES CXX C Fortran)
```

2. 使用C++11标准:

```cmake
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```











