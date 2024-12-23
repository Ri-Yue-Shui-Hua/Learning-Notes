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

3. 此外，我们验证Fortran和C/C++编译器是否能协同工作，并生成头文件，这个文件可以处理名称混乱。两个功能都由`FortranCInterface`模块提供:

```cmake
include(FortranCInterface)

FortranCInterface_VERIFY(CXX)

FortranCInterface_HEADER(
  fc_mangle.h
  MACRO_NAMESPACE "FC_"
  SYMBOLS DSCAL DGESV
  )
```

4. 然后，找到BLAS和LAPACK:

```cmake
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
```

5. 接下来，添加一个库，其中包含BLAS和LAPACK包装器的源代码，并链接到`LAPACK_LIBRARIES`，其中也包含`BLAS_LIBRARIES`:

```cmake
add_library(math "")

target_sources(math
  PRIVATE
    CxxBLAS.cpp
    CxxLAPACK.cpp
  )
  
target_include_directories(math
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
  )
  
target_link_libraries(math
  PUBLIC
  	${LAPACK_LIBRARIES}
  )
```

6. 注意，目标的包含目录和链接库声明为`PUBLIC`，因此任何依赖于数学库的附加目标也将在其包含目录中。
7. 最后，我们添加一个可执行目标并链接`math`：

```cmake
add_executable(linear-algebra "")

target_sources(linear-algebra
  PRIVATE
  	linear-algebra.cpp
  )

target_link_libraries(linear-algebra
  PRIVATE
  	math
  )
```

8. 配置时，我们可以关注相关的打印输出:

```bash
$ mkdir -p build
$ cd build
$ cmake ..

...
-- Detecting Fortran/C Interface
-- Detecting Fortran/C Interface - Found GLOBAL and MODULE mangling
-- Verifying Fortran/C Compiler Compatibility
-- Verifying Fortran/C Compiler Compatibility - Success
...
-- Found BLAS: /usr/lib/libblas.so
...
-- A library with LAPACK API found.
...
```

9. 最后，构建并测试可执行文件:

```bash
$ cmake --build .
$ ./linear-algebra 1000

C_DSCAL done
C_DGESV done
info is 0
check is 1.54284e-10
```

### 工作原理

`FindBLAS.cmake`和`FindLAPACK.cmake`将在标准位置查找BLAS和LAPACK库。对于前者，该模块有`SGEMM`函数的Fortran实现，一般用于单精度矩阵乘积。对于后者，该模块有`CHEEV`函数的Fortran实现，用于计算复杂厄米矩阵的特征值和特征向量。查找在CMake内部，通过编译一个小程序来完成，该程序调用这些函数，并尝试链接到候选库。如果失败，则表示相应库不存于系统上。

生成机器码时，每个编译器都会处理符号混淆，不幸的是，这种操作并不通用，而与编译器相关。为了解决这个问题，我们使用`FortranCInterface`模块( https://cmake.org/cmake/help/v3.5/module/FortranCInterface.html )验证Fortran和C/C++能否混合编译，然后生成一个Fortran-C接口头文件`fc_mangle.h`，这个文件用来解决编译器性的问题。然后，必须将生成的`fc_mann .h`包含在接口头文件`CxxBLAS.hpp`和`CxxLAPACK.hpp`中。为了使用`FortranCInterface`，我们需要在`LANGUAGES`列表中添加C和Fortran支持。当然，也可以定义自己的预处理器定义，但是可移植性会差很多。

我们将在第9章中更详细地讨论Fortran和C的互操作性。

**NOTE**:*目前，BLAS和LAPACK的许多实现已经在Fortran外附带了一层C包装。这些包装器多年来已经标准化，称为CBLAS和LAPACKE。*

### 更多信息

许多算法代码比较依赖于矩阵代数运算，使用BLAS和LAPACK API的高性能实现就非常重要了。供应商为不同的体系结构和并行环境提供不同的库，`FindBLAS.cmake`和`FindLAPACK.cmake`可能的无法定位到当前库。如果发生这种情况，可以通过`-D`选项显式地从CLI对库进行设置。

## 检测OpenMP的并行环境

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-03/recipe-05 中找到，有一个C++和一个Fortran示例。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-03/recipe-05 中也有一个适用于CMake 3.5的示例。*

目前，市面上的计算机几乎都是多核机器，对于性能敏感的程序，我们必须关注这些多核处理器，并在编程模型中使用并发。OpenMP是多核处理器上并行性的标准之一。为了从OpenMP并行化中获得性能收益，通常不需要修改或重写现有程序。一旦确定了代码中的性能关键部分，例如：使用分析工具，程序员就可以通过预处理器指令，指示编译器为这些区域生成可并行的代码。

本示例中，我们将展示如何编译一个包含OpenMP指令的程序(前提是使用一个支持OpenMP的编译器)。有许多支持OpenMP的Fortran、C和C++编译器。对于相对较新的CMake版本，为OpenMP提供了非常好的支持。本示例将展示如何在使用CMake 3.9或更高版本时，使用简单C++和Fortran程序来链接到OpenMP。

**NOTE**:*根据Linux发行版的不同，Clang编译器的默认版本可能不支持OpenMP。使用或非苹果版本的Clang(例如，Conda提供的)或GNU编译器,除非单独安装libomp库(https://iscinumpy.gitlab.io/post/omp-on-high-sierra/ )，否则本节示例将无法在macOS上工作。*

### 准备工作

C和C++程序可以通过包含`omp.h`头文件和链接到正确的库，来使用OpenMP功能。编译器将在性能关键部分之前添加预处理指令，并生成并行代码。在本示例中，我们将构建以下示例源代码(`example.cpp`)。这段代码从1到N求和，其中N作为命令行参数:

```cmake
#include <iostream>
#include <omp.h>
#include <string>

int main(int argc, char *argv[])
{
  std::cout << "number of available processors: " << omp_get_num_procs()
            << std::endl;
  std::cout << "number of threads: " << omp_get_max_threads() << std::endl;
  auto n = std::stol(argv[1]);
  std::cout << "we will form sum of numbers from 1 to " << n << std::endl;
  // start timer
  auto t0 = omp_get_wtime();
  auto s = 0LL;
#pragma omp parallel for reduction(+ : s)
  for (auto i = 1; i <= n; i++)
  {
    s += i;
  }
  // stop timer
  auto t1 = omp_get_wtime();

  std::cout << "sum: " << s << std::endl;
  std::cout << "elapsed wall clock time: " << t1 - t0 << " seconds" << std::endl;
  
  return 0;
}
```

在Fortran语言中，需要使用`omp_lib`模块并链接到库。在性能关键部分之前的代码注释中，可以再次使用并行指令。例如：`F90`需要包含以下内容:



```fortran
program example

  use omp_lib
  
  implicit none
  
  integer(8) :: i, n, s
  character(len=32) :: arg
  real(8) :: t0, t1
  
  print *, "number of available processors:", omp_get_num_procs()
  print *, "number of threads:", omp_get_max_threads()
  
  call get_command_argument(1, arg)
  read(arg , *) n
  
  print *, "we will form sum of numbers from 1 to", n
  
  ! start timer
  t0 = omp_get_wtime()
  
  s = 0
!$omp parallel do reduction(+:s)
  do i = 1, n
  s = s + i
  end do
  
  ! stop timer
  t1 = omp_get_wtime()
  
  print *, "sum:", s
  print *, "elapsed wall clock time (seconds):", t1 - t0
  
end program
```

### 具体实施

对于C++和Fortran的例子，`CMakeLists.txt`将遵循一个模板，该模板在这两种语言上很相似：

1. 两者都定义了CMake最低版本、项目名称和语言(CXX或Fortran；我们将展示C++版本):

```cmake
cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(recipe-05 LANGUAGES CXX)
```

2. 使用C++11标准:

```cmake
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

3. 调用find_package来搜索OpenMP:

```cmake
find_package(OpenMP REQUIRED)
```

4. 最后，我们定义可执行目标，并链接到FindOpenMP模块提供的导入目标(在Fortran的情况下，我们链接到`OpenMP::OpenMP_Fortran`):

```cmake
add_executable(example example.cpp)
target_link_libraries(example
  PUBLIC
  	OpenMP::OpenMP_CXX
  )
```

5. 现在，可以配置和构建代码了:

```bash
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
```

6. 并行测试(在本例中使用了4个内核):

```bash
$ ./example 1000000000

number of available processors: 4
number of threads: 4
we will form sum of numbers from 1 to 1000000000
sum: 500000000500000000
elapsed wall clock time: 1.08343 seconds
```

7. 为了比较，我们可以重新运行这个例子，并将OpenMP线程的数量设置为1:

```bash
$ env OMP_NUM_THREADS=1 ./example 1000000000

number of available processors: 4
number of threads: 1
we will form sum of numbers from 1 to 1000000000
sum: 500000000500000000
elapsed wall clock time: 2.96427 seconds

```

### 工作原理

我们的示例很简单：编译代码，并运行在多个内核上时，我们会看到加速效果。加速效果并不是`OMP_NUM_THREADS`的倍数，不过本示例中并不关心，因为我们更关注的是如何使用CMake配置需要使用OpenMP的项目。我们发现链接到OpenMP非常简单，这要感谢`FindOpenMP`模块:

```cmake
target_link_libraries(example
	PUBLIC
		OpenMP::OpenMP_CXX
	)
```

我们不关心编译标志或包含目录——这些设置和依赖项是在`OpenMP::OpenMP_CXX`中定义的(`IMPORTED`类型)。如第1章第3节中提到的，`IMPORTED`库是伪目标，它完全是我们自己项目的外部依赖项。要使用OpenMP，需要设置一些编译器标志，包括目录和链接库。所有这些都包含在`OpenMP::OpenMP_CXX`的属性上，并通过使用`target_link_libraries`命令传递给`example`。这使得在CMake中，使用库变得非常容易。我们可以使用`cmake_print_properties`命令打印接口的属性，该命令由`CMakePrintHelpers.CMake`模块提供:

```cmake
include(CMakePrintHelpers)
cmake_print_properties(
	TARGETS
		OpenMP::OpenMP_CXX
	PROPERTIES
		INTERFACE_COMPILE_OPTIONS
		INTERFACE_INCLUDE_DIRECTORIES
		INTERFACE_LINK_LIBRARIES
	)
```

所有属性都有`INTERFACE_`前缀，因为这些属性对所需目标，需要以接口形式提供，并且目标以接口的方式使用OpenMP。

对于低于3.9的CMake版本:

```cmake
add_executable(example example.cpp)

target_compile_options(example
  PUBLIC
  	${OpenMP_CXX_FLAGS}
  )
  
set_target_properties(example
  PROPERTIES
  	LINK_FLAGS ${OpenMP_CXX_FLAGS}
  )
```

对于低于3.5的CMake版本，我们需要为Fortran项目显式定义编译标志。

在这个示例中，我们讨论了C++和Fortran。相同的参数和方法对于C项目也有效。

## 检测MPI的并行环境

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-03/recipe-06 中找到，包含一个C++和一个C的示例。该示例在CMake 3.9版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-03/recipe-06 中也有一个适用于CMake 3.5的C示例。*

消息传递接口(Message Passing Interface, MPI)，可以作为OpenMP(共享内存并行方式)的补充，它也是分布式系统上并行程序的实际标准。尽管，最新的MPI实现也允许共享内存并行，但高性能计算中的一种典型方法就是，在计算节点上OpenMP与MPI结合使用。MPI标准的实施包括:

1. 运行时库
2. 头文件和Fortran 90模块
3. 编译器的包装器，用来调用编译器，使用额外的参数来构建MPI库，以处理目录和库。通常，包装器`mpic++/mpiCC/mpicxx`用于C++，`mpicc`用于C，`mpifort`用于Fortran。
4. 启动MPI：应该启动程序，以编译代码的并行执行。它的名称依赖于实现，可以使用这几个命令启动：`mpirun`、`mpiexec`或`orterun`。

本示例，将展示如何在系统上找到合适的MPI实现，从而编译一个简单的“Hello, World”MPI例程。

### 准备工作

示例代码(`hello-mpi.cpp`，可从[http://www.mpitutorial.com](http://www.mpitutorial.com/) 下载)将在本示例中进行编译，它将初始化MPI库，让每个进程打印其名称:

```c++
#include <iostream>

#include <mpi.h>

int main(int argc, char **argv)
{
  // Initialize the MPI environment. The two arguments to MPI Init are not
  // currently used by MPI implementations, but are there in case future
  // implementations might need the arguments.
  MPI_Init(NULL, NULL);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  // Print off a hello world message
  std::cout << "Hello world from processor " << processor_name << ", rank "
            << world_rank << " out of " << world_size << " processors" << std::endl;
            
  // Finalize the MPI environment. No more MPI calls can be made after this
  MPI_Finalize();
}
```

### 具体实施

这个示例中，我们先查找MPI实现：库、头文件、编译器包装器和启动器。为此，我们将用到`FindMPI.cmake`标准CMake模块:

1. 首先，定义了CMake最低版本、项目名称、支持的语言和语言标准:

```cmake
cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(recipe-06 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

2. 然后，调用`find_package`来定位MPI:

```cmake
find_package(MPI REQUIRED)
```

3. 与前面的配置类似，定义了可执行文件的的名称和相关源码，并链接到目标:

```cmake
add_executable(hello-mpi hello-mpi.cpp)

target_link_libraries(hello-mpi
  PUBLIC
 	  MPI::MPI_CXX
  )
```

4. 配置和构建可执行文件:

```bash
$ mkdir -p build
$ cd build
$ cmake .. # -D CMAKE_CXX_COMPILER=mpicxx C++例子中可加，加与不加对于构建结果没有影响╭(╯^╰)╮

-- ...
-- Found MPI_CXX: /usr/lib/openmpi/libmpi_cxx.so (found version "3.1")
-- Found MPI: TRUE (found version "3.1")
-- ...

$ cmake --build .
```

5. 为了并行执行这个程序，我们使用`mpirun`启动器(本例中，启动了两个任务):

```bash
$ mpirun -np 2 ./hello-mpi

Hello world from processor larry, rank 1 out of 2 processors
Hello world from processor larry, rank 0 out of 2 processors

```

### 工作原理

请记住，编译包装器是对MPI库编译器的封装。底层实现中，将会调用相同的编译器，并使用额外的参数(如成功构建并行程序所需的头文件包含路径和库)来扩充它。

编译和链接源文件时，包装器用了哪些标志？我们可以使用`--showme`选项来查看。要找出编译器的标志，我们可以这样使用:

```bash
$ mpicxx --showme:compile

-pthread
```

为了找出链接器标志，我们可以这样:

```bash
$ mpicxx --showme:link

-pthread -Wl,-rpath -Wl,/usr/lib/openmpi -Wl,--enable-new-dtags -L/usr/lib/openmpi -lmpi_cxx -lmpi
```

与之前的OpenMP配置类似，我们发现到MPI的链接非常简单，这要归功于`FindMPI`模块提供的目标:

正如在前面的配方中所讨论的，对于CMake版本低于3.9，需要更多的工作量:

```cmake
add_executable(hello-mpi hello-mpi.c)

target_compile_options(hello-mpi
  PUBLIC
  	${MPI_CXX_COMPILE_FLAGS}
  )
  
target_include_directories(hello-mpi
  PUBLIC
  	${MPI_CXX_INCLUDE_PATH}
  )
  
target_link_libraries(hello-mpi
  PUBLIC
  	${MPI_CXX_LIBRARIES}
  )
```

本示例中，我们讨论了C++项目。其中的参数和方法对于C或Fortran项目同样有效。

## 检测Eigen库

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-03/recipe-07 中找到，包含一个C++的示例。该示例在CMake 3.9版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-03/recipe-06 中也有一个适用于CMake 3.5的C++示例。*

BLAS库为矩阵和向量操作提供了标准化接口。不过，这个接口用Fortran语言书写。虽然已经展示了如何使用C++直接使用这些库，但在现代C++程序中，希望有更高级的接口。

纯头文件实现的Eigen库，使用模板编程来提供接口。矩阵和向量的计算，会在编译时进行数据类型检查，以确保兼容所有维度的矩阵。密集和稀疏矩阵的运算，也可使用表达式模板高效的进行实现，如：矩阵-矩阵乘积，线性系统求解器和特征值问题。从3.3版开始，Eigen可以链接到BLAS和LAPACK库中，这可以将某些操作实现进行卸载，使库的实现更加灵活，从而获得更多的性能收益。

本示例将展示如何查找Eigen库，使用OpenMP并行化，并将部分工作转移到BLAS库。

本示例中会实现，矩阵-向量乘法和LU分解，可以选择卸载BLAS和LAPACK库中的一些实现。这个示例中，只考虑将在BLAS库中卸载。

### 准备工作

本例中，我们编译一个程序，该程序会从命令行获取的随机方阵和维向量。然后我们将用LU分解来解线性方程组**Ax=b**。以下是源代码(`linear-algebra.cpp`):

```c++
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include <Eigen/Dense>

int main(int argc, char **argv)
{
  if (argc != 2)
  {
    std::cout << "Usage: ./linear-algebra dim" << std::endl;
    return EXIT_FAILURE;
  }
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> elapsed_seconds;
  std::time_t end_time;
  std::cout << "Number of threads used by Eigen: " << Eigen::nbThreads()
            << std::endl;

  // Allocate matrices and right-hand side vector
  start = std::chrono::system_clock::now();
  int dim = std::atoi(argv[1]);
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(dim, dim);
  Eigen::VectorXd b = Eigen::VectorXd::Random(dim);
  end = std::chrono::system_clock::now();

  // Report times
  elapsed_seconds = end - start;
  end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "matrices allocated and initialized "
            << std::put_time(std::localtime(&end_time), "%a %b %d %Y
  %r\n")
            << "elapsed time: " << elapsed_seconds.count() << "s\n";

  start = std::chrono::system_clock::now();
  // Save matrix and RHS
  Eigen::MatrixXd A1 = A;
  Eigen::VectorXd b1 = b;
  end = std::chrono::system_clock::now();
  end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "Scaling done, A and b saved "
            << std::put_time(std::localtime(&end_time), "%a %b %d %Y %r\n")
            << "elapsed time: " << elapsed_seconds.count() << "s\n";
  start = std::chrono::system_clock::now();
  Eigen::VectorXd x = A.lu().solve(b);
  end = std::chrono::system_clock::now();

  // Report times
  elapsed_seconds = end - start;
  end_time = std::chrono::system_clock::to_time_t(end);
  double relative_error = (A * x - b).norm() / b.norm();
  std::cout << "Linear system solver done "
            << std::put_time(std::localtime(&end_time), "%a %b %d %Y %r\n")
            << "elapsed time: " << elapsed_seconds.count() << "s\n";
  std::cout << "relative error is " << relative_error << std::endl;
  
  return 0;
}
```

矩阵-向量乘法和LU分解是在Eigen库中实现的，但是可以选择BLAS和LAPACK库中的实现。在这个示例中，我们只考虑BLAS库中的实现。

### 具体实施

这个示例中，我们将用到Eigen和BLAS库，以及OpenMP。使用OpenMP将Eigen并行化，并从BLAS库中卸载部分线性代数实现:

1. 首先声明CMake最低版本、项目名称和使用C++11语言标准:

```cmake
cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(recipe-07 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

2. 因为Eigen可以使用共享内存的方式，所以可以使用OpenMP并行处理计算密集型操作:

```cmake
find_package(OpenMP REQUIRED)
```

3. 调用`find_package`来搜索Eigen(将在下一小节中讨论):

```cmake
find_package(Eigen3 3.3 REQUIRED CONFIG)
```

4. 如果找到Eigen，我们将打印状态信息。注意，使用的是`Eigen3::Eigen`，这是一个`IMPORT`目标，可通过提供的CMake脚本找到这个目标:

```cmake
if(TARGET Eigen3::Eigen)
  message(STATUS "Eigen3 v${EIGEN3_VERSION_STRING} found in ${EIGEN3_INCLUDE_DIR}")
endif()
```

5. 接下来，将源文件声明为可执行目标:

```cmake
add_executable(linear-algebra linear-algebra.cpp)
```

6. 然后，找到BLAS。注意，现在不需要依赖项:

```cmake
find_package(BLAS)
```

7. 如果找到BLAS，我们可为可执行目标，设置相应的宏定义和链接库:

```cmake
if(BLAS_FOUND)
  message(STATUS "Eigen will use some subroutines from BLAS.")
  message(STATUS "See: http://eigen.tuxfamily.org/dox-devel/TopicUsingBlasLapack.html")
  target_compile_definitions(linear-algebra
    PRIVATE
    	EIGEN_USE_BLAS
    )
  target_link_libraries(linear-algebra
    PUBLIC
    	${BLAS_LIBRARIES}
    )
else()
	message(STATUS "BLAS not found. Using Eigen own functions")
endif()
```

8. 最后，我们链接到`Eigen3::Eigen`和`OpenMP::OpenMP_CXX`目标。这就可以设置所有必要的编译标示和链接标志:

```cmake
target_link_libraries(linear-algebra
  PUBLIC
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
  )
```

9. 开始配置:

```bash
$ mkdir -p build
$ cd build
$ cmake ..

-- ...
-- Found OpenMP_CXX: -fopenmp (found version "4.5")
-- Found OpenMP: TRUE (found version "4.5")
-- Eigen3 v3.3.4 found in /usr/include/eigen3
-- ...
-- Found BLAS: /usr/lib/libblas.so
-- Eigen will use some subroutines from BLAS.
-- See: http://eigen.tuxfamily.org/dox-devel/TopicUsingBlasLapack.html

```

10. 最后，编译并测试代码。注意，可执行文件使用四个线程运行:

```bash
$ cmake --build .
$ ./linear-algebra 1000

Number of threads used by Eigen: 4
matrices allocated and initialized Sun Jun 17 2018 11:04:20 AM
elapsed time: 0.0492328s
Scaling done, A and b saved Sun Jun 17 2018 11:04:20 AM
elapsed time: 0.0492328s
Linear system solver done Sun Jun 17 2018 11:04:20 AM
elapsed time: 0.483142s
relative error is 4.21946e-13
```

### 工作原理

Eigen支持CMake查找，这样配置项目就会变得很容易。从3.3版开始，Eigen提供了CMake模块，这些模块将导出相应的目标`Eigen3::Eigen`。

`find_package`可以通过选项传递，届时CMake将不会使用`FindEigen3.cmake`模块，而是通过特定的`Eigen3Config.cmake`，`Eigen3ConfigVersion.cmake`和`Eigen3Targets.cmake`提供Eigen3安装的标准位置(`<installation-prefix>/share/eigen3/cmake`)。这种包定位模式称为“Config”模式，比`Find<package>.cmake`方式更加通用。有关“模块”模式和“配置”模式的更多信息，可参考官方文档 https://cmake.org/cmake/help/v3.5/command/find_package.html 。

虽然Eigen3、BLAS和OpenMP声明为`PUBLIC`依赖项，但`EIGEN_USE_BLAS`编译定义声明为`PRIVATE`。可以在单独的库目标中汇集库依赖项，而不是直接链接可执行文件。使用`PUBLIC/PRIVATE`关键字，可以根据库目标的依赖关系调整相应标志和定义。

### 更多信息

CMake将在预定义的位置层次结构中查找配置模块。首先是`CMAKE_PREFIX_PATH`，`<package>_DIR`是接下来的搜索路径。因此，如果Eigen3安装在非标准位置，可以使用这两个选项来告诉CMake在哪里查找它:

1. 通过将Eigen3的安装前缀传递给`CMAKE_PREFIX_PATH`:

```bash
$ cmake -D CMAKE_PREFIX_PATH=<installation-prefix> ..
```

2. 通过传递配置文件的位置作为`Eigen3_DIR`:

```bash
$ cmake -D Eigen3_DIR=<installation-prefix>/share/eigen3/cmake ..
```

## 检测Boost库

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-03/recipe-08 中找到，包含一个C++的示例。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

Boost是一组C++通用库。这些库提供了许多功能，这些功能在现代C++项目中不可或缺，但是还不能通过C++标准使用这些功能。例如，Boost为元编程、处理可选参数和文件系统操作等提供了相应的组件。这些库中有许多特性后来被C++11、C++14和C++17标准所采用，但是对于保持与旧编译器兼容性的代码库来说，许多Boost组件仍然是首选。

本示例将向您展示如何检测和链接Boost库的一些组件。

### 准备工作

我们将编译的源码是Boost提供的文件系统库与文件系统交互的示例。这个库可以跨平台使用，并将操作系统和文件系统之间的差异抽象为一致的API。下面的代码(`path-info.cpp`)将接受一个路径作为参数，并将其组件的报告打印到屏幕上:

```c++
#include <iostream>

#include <boost/filesystem.hpp>

using namespace std;
using namespace boost::filesystem;
const char *say_what(bool b) { return b ? "true" : "false"; }
int main(int argc, char *argv[])
{
  if (argc < 2)
  {
    cout
        << "Usage: path_info path-element [path-element...]\n"
           "Composes a path via operator/= from one or more path-element arguments\n"
           "Example: path_info foo/bar baz\n"
#ifdef BOOST_POSIX_API
           " would report info about the composed path foo/bar/baz\n";
#else // BOOST_WINDOWS_API
           " would report info about the composed path foo/bar\\baz\n";
#endif
    return 1;
  }
  path p;
  for (; argc > 1; --argc, ++argv)
    p /= argv[1]; // compose path p from the command line arguments
  cout << "\ncomposed path:\n";
  cout << " operator<<()---------: " << p << "\n";
  cout << " make_preferred()-----: " << p.make_preferred() << "\n";
  cout << "\nelements:\n";
  for (auto element : p)
    cout << " " << element << '\n';
  cout << "\nobservers, native format:" << endl;
#ifdef BOOST_POSIX_API
  cout << " native()-------------: " << p.native() << endl;
  cout << " c_str()--------------: " << p.c_str() << endl;
#else // BOOST_WINDOWS_API
  wcout << L" native()-------------: " << p.native() << endl;
  wcout << L" c_str()--------------: " << p.c_str() << endl;
#endif
  cout << " string()-------------: " << p.string() << endl;
  wcout << L" wstring()------------: " << p.wstring() << endl;
  cout << "\nobservers, generic format:\n";
  cout << " generic_string()-----: " << p.generic_string() << endl;
  wcout << L" generic_wstring()----: " << p.generic_wstring() << endl;
  cout << "\ndecomposition:\n";
  cout << " root_name()----------: " << p.root_name() << '\n';
  cout << " root_directory()-----: " << p.root_directory() << '\n';
  cout << " root_path()----------: " << p.root_path() << '\n';
  cout << " relative_path()------: " << p.relative_path() << '\n';
  cout << " parent_path()--------: " << p.parent_path() << '\n';
  cout << " filename()-----------: " << p.filename() << '\n';
  cout << " stem()---------------: " << p.stem() << '\n';
  cout << " extension()----------: " << p.extension() << '\n';
  cout << "\nquery:\n";
  cout << " empty()--------------: " << say_what(p.empty()) << '\n';
  cout << " is_absolute()--------: " << say_what(p.is_absolute()) << '\n';
  cout << " has_root_name()------: " << say_what(p.has_root_name()) << '\n';
  cout << " has_root_directory()-: " << say_what(p.has_root_directory()) << '\n';
  cout << " has_root_path()------: " << say_what(p.has_root_path()) << '\n';
  cout << " has_relative_path()--: " << say_what(p.has_relative_path()) << '\n';
  cout << " has_parent_path()----: " << say_what(p.has_parent_path()) << '\n';
  cout << " has_filename()-------: " << say_what(p.has_filename()) << '\n';
  cout << " has_stem()-----------: " << say_what(p.has_stem()) << '\n';
  cout << " has_extension()------: " << say_what(p.has_extension()) << '\n';
  return 0;
}
```

### 具体实施

Boost由许多不同的库组成，这些库可以独立使用。CMake可将这个库集合，表示为组件的集合。`FindBoost.cmake`模块不仅可以搜索库集合的完整安装，还可以搜索集合中的特定组件及其依赖项(如果有的话)。我们将逐步建立相应的`CMakeLists.txt`:

1. 首先，声明CMake最低版本、项目名称、语言，并使用C++11标准:

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-08 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

2. 然后，使用`find_package`搜索Boost。若需要对Boost强制性依赖，需要一个参数。这个例子中，只需要文件系统组件，所以将它作为参数传递给`find_package`:

```cmake
find_package(Boost 1.54 REQUIRED COMPONENTS filesystem)
```

3. 添加可执行目标，编译源文件:

```cmake
add_executable(path-info path-info.cpp)
```

4. 最后，将目标链接到Boost库组件。由于依赖项声明为`PUBLIC`，依赖于Boost的目标将自动获取依赖项:

```cmake
target_link_libraries(path-info
  PUBLIC
  	Boost::filesystem
	)
```

### 工作原理

`FindBoost.cmake`是本示例中所使用的CMake模块，其会在标准系统安装目录中找到Boost库。由于我们链接的是`Boost::filesystem`，CMake将自动设置包含目录并调整编译和链接标志。如果Boost库安装在非标准位置，可以在配置时使用`BOOST_ROOT`变量传递Boost安装的根目录，以便让CMake搜索非标准路径:

```bash
$ cmake -D BOOST_ROOT=/custom/boost
```

或者，可以同时传递包含头文件的`BOOST_INCLUDEDIR`变量和库目录的`BOOST_LIBRARYDIR`变量:

```bash
$ cmake -D BOOST_INCLUDEDIR=/custom/boost/include -DBOOST_LIBRARYDIR=/custom/boost/lib
```

## 检测外部库:Ⅰ. 使用pkg-config

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-03/recipe-09 中找到，包含一个C的示例。该示例在CMake 3.6版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-03/recipe-09 中也有一个适用于CMake 3.5的示例。*

目前为止，我们已经讨论了两种检测外部依赖关系的方法:

- 使用CMake自带的`find-module`，但并不是所有的包在CMake的`find`模块都找得到。
- 使用`<package>Config.cmake`, `<package>ConfigVersion.cmake`和`<package>Targets.cmake`，这些文件由软件包供应商提供，并与软件包一起安装在标准位置的cmake文件夹下。

如果某个依赖项既不提供查找模块，也不提供供应商打包的CMake文件，该怎么办?在这种情况下，我们只有两个选择:

- 依赖`pkg-config`程序，来找到系统上的包。这依赖于包供应商在`.pc`配置文件中，其中有关于发行包的元数据。
- 为依赖项编写自己的`find-package`模块。

本示例中，将展示如何利用CMake中的`pkg-config`来定位ZeroMQ消息库。下一个示例中，将编写一个find模块，展示如何为ZeroMQ编写属于自己`find`模块。

### 准备工作

我们构建的代码来自ZeroMQ手册 http://zguide.zeromq.org/page:all 的示例。由两个源文件`hwserver.c`和`hwclient.c`组成，这两个源文件将构建为两个独立的可执行文件。执行时，它们将打印“Hello, World”。

### 具体实施

这是一个C项目，我们将使用C99标准，逐步构建`CMakeLists.txt`文件:

1. 声明一个C项目，并要求符合C99标准:

```cmake
cmake_minimum_required(VERSION 3.6 FATAL_ERROR)

project(recipe-09 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_C_STANDARD_REQUIRED ON)
```

2. 使用CMake附带的find-module，查找`pkg-config`。这里在`find_package`中传递了`QUIET`参数。只有在没有找到`pkg-config`时，CMake才会报错:

```cmake
find_package(PkgConfig REQUIRED QUIET)
```

3. 找到`pkg-config`时，我们将使用`pkg_search_module`函数，以搜索任何附带包配置`.pc`文件的库或程序。该示例中，我们查找ZeroMQ库:

```cmake
pkg_search_module(
  ZeroMQ
  REQUIRED
  	libzeromq libzmq lib0mq
  IMPORTED_TARGET
  )
```

4. 如果找到ZeroMQ库，则打印状态消息:

```cmake
if(TARGET PkgConfig::ZeroMQ)
	message(STATUS "Found ZeroMQ")
endif()
```

5. 然后，添加两个可执行目标，并链接到ZeroMQ。这将自动设置包括目录和链接库:

```cmake
add_executable(hwserver hwserver.c)
target_link_libraries(hwserver PkgConfig::ZeroMQ)
add_executable(hwclient hwclient.c)
target_link_libraries(hwclient PkgConfig::ZeroMQ)
```

6. 现在，我们可以配置和构建示例:

```bash
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
```

7. 在终端中，启动服务器，启动时会输出类似于本例的消息:

```bash
Current 0MQ version is 4.2.2
```

8. 然后，在另一个终端启动客户端，它将打印如下内容:

```bash
Connecting to hello world server…
Sending Hello 0…
Received World 0
Sending Hello 1…
Received World 1
Sending Hello 2…
...
```

### 工作原理

当找到`pkg-config`时, CMake需要提供两个函数，来封装这个程序提供的功能:

- `pkg_check_modules`，查找传递列表中的所有模块(库和/或程序)
- `pkg_search_module`，要在传递的列表中找到第一个工作模块

与`find_package`一样，这些函数接受`REQUIRED`和`QUIET`参数。更详细地说，我们对`pkg_search_module`的调用如下:

```cmake
pkg_search_module(
  ZeroMQ
  REQUIRED
  	libzeromq libzmq lib0mq
  IMPORTED_TARGET
  )
```

这里，第一个参数是前缀，它将用于命名存储搜索ZeroMQ库结果的目标：`PkgConfig::ZeroMQ`。注意，我们需要为系统上的库名传递不同的选项：`libzeromq`、`libzmq`和`lib0mq`。这是因为不同的操作系统和包管理器，可为同一个包选择不同的名称。

**NOTE**:*`pkg_check_modules`和`pkg_search_module`函数添加了`IMPORTED_TARGET`选项，并在CMake 3.6中定义导入目标的功能。3.6之前的版本，只定义了变量`ZeroMQ_INCLUDE_DIRS`(用于include目录)和`ZeroMQ_LIBRARIES`(用于链接库)，供后续使用。*

## 检测外部库:Ⅱ. 自定义find模块

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-03/recipe-10 中找到，包含一个C的示例。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

此示例补充了上一节的示例，我们将展示如何编写一个`find`模块来定位系统上的ZeroMQ消息库，以便能够在非Unix操作系统上检测该库。我们重用服务器-客户端示例代码。

### 如何实施

这是一个C项目，使用C99标准，并逐步构建CMakeLists.txt文件:

1. 声明一个C项目，并要求符合C99标准:

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-10 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_C_STANDARD_REQUIRED ON)
```

2. 将当前源目录`CMAKE_CURRENT_SOURCE_DIR`，添加到CMake将查找模块的路径列表`CMAKE_MODULE_PATH`中。这样CMake就可以找到，我们自定义的`FindZeroMQ.cmake`模块:

```cmake
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
```

3. 现在`FindZeroMQ.cmake`模块是可用的，可以通过这个模块来搜索项目所需的依赖项。由于我们没有使用`QUIET`选项来查找`find_package`，所以当找到库时，状态消息将自动打印:

```cmake
find_package(ZeroMQ REQUIRED)
```

4. 我们继续添加`hwserver`可执行目标。头文件包含目录和链接库是使用`find_package`命令成功后，使用`ZeroMQ_INCLUDE_DIRS`和`ZeroMQ_LIBRARIES`变量进行指定的:

```cmake
add_executable(hwserver hwserver.c)
target_include_directories(hwserver
  PRIVATE
  	${ZeroMQ_INCLUDE_DIRS}
  )
target_link_libraries(hwserver
  PRIVATE
  	${ZeroMQ_LIBRARIES}
  )
```

5. 最后，我们对`hwclient`可执行目标执行相同的操作:

```cmake
add_executable(hwclient hwclient.c)
target_include_directories(hwclient
  PRIVATE
  	${ZeroMQ_INCLUDE_DIRS}
  )
target_link_libraries(hwclient
  PRIVATE
  	${ZeroMQ_LIBRARIES}
  )
```

此示例的主`CMakeLists.txt`在使用`FindZeroMQ.cmake`时，与前一个示例中使用的`CMakeLists.txt`不同。这个模块使用`find_path`和`find_library` CMake内置命令，搜索ZeroMQ头文件和库，并使用`find_package_handle_standard_args`设置相关变量，就像我们在第3节中做的那样。

1. `FindZeroMQ.cmake`中，检查了`ZeroMQ_ROOT`变量是否设置。此变量可用于ZeroMQ库的检测，并引导到自定义安装目录。用户可能设置了`ZeroMQ_ROOT`作为环境变量，我们也会进行检查了:

```cmake
if(NOT ZeroMQ_ROOT)
	set(ZeroMQ_ROOT "$ENV{ZeroMQ_ROOT}")
endif()
```

2. 然后，搜索系统上`zmq.h`头文件的位置。这是基于`_ZeroMQ_ROOT`变量和`find_path`命令进行的:

```cmake
if(NOT ZeroMQ_ROOT)
	find_path(_ZeroMQ_ROOT NAMES include/zmq.h)
else()
	set(_ZeroMQ_ROOT "${ZeroMQ_ROOT}")
endif()

find_path(ZeroMQ_INCLUDE_DIRS NAMES zmq.h HINTS ${_ZeroMQ_ROOT}/include)
```

3. 如果成功找到头文件，则将`ZeroMQ_INCLUDE_DIRS`设置为其位置。我们继续通过使用字符串操作和正则表达式，寻找相应版本的ZeroMQ库:

```cmake
set(_ZeroMQ_H ${ZeroMQ_INCLUDE_DIRS}/zmq.h)

function(_zmqver_EXTRACT _ZeroMQ_VER_COMPONENT _ZeroMQ_VER_OUTPUT)
set(CMAKE_MATCH_1 "0")
set(_ZeroMQ_expr "^[ \\t]*#define[ \\t]+${_ZeroMQ_VER_COMPONENT}[ \\t]+([0-9]+)$")
file(STRINGS "${_ZeroMQ_H}" _ZeroMQ_ver REGEX "${_ZeroMQ_expr}")
string(REGEX MATCH "${_ZeroMQ_expr}" ZeroMQ_ver "${_ZeroMQ_ver}")
set(${_ZeroMQ_VER_OUTPUT} "${CMAKE_MATCH_1}" PARENT_SCOPE)
endfunction()

_zmqver_EXTRACT("ZMQ_VERSION_MAJOR" ZeroMQ_VERSION_MAJOR)
_zmqver_EXTRACT("ZMQ_VERSION_MINOR" ZeroMQ_VERSION_MINOR)
_zmqver_EXTRACT("ZMQ_VERSION_PATCH" ZeroMQ_VERSION_PATCH)
```

4. 然后，为`find_package_handle_standard_args`准备`ZeroMQ_VERSION`变量:

```cmake
if(ZeroMQ_FIND_VERSION_COUNT GREATER 2)
	set(ZeroMQ_VERSION "${ZeroMQ_VERSION_MAJOR}.${ZeroMQ_VERSION_MINOR}.${ZeroMQ_VERSION_PATCH}")
else()
	set(ZeroMQ_VERSION "${ZeroMQ_VERSION_MAJOR}.${ZeroMQ_VERSION_MINOR}")
endif()
```

5. 使用`find_library`命令搜索ZeroMQ库。因为库的命名有所不同，这里我们需要区分Unix的平台和Windows平台:

```cmake
if(NOT ${CMAKE_C_PLATFORM_ID} STREQUAL "Windows")
  find_library(ZeroMQ_LIBRARIES
    NAMES
    	zmq
    HINTS
      ${_ZeroMQ_ROOT}/lib
      ${_ZeroMQ_ROOT}/lib/x86_64-linux-gnu
    )
else()
  find_library(ZeroMQ_LIBRARIES
    NAMES
    	libzmq
      "libzmq-mt-${ZeroMQ_VERSION_MAJOR}_${ZeroMQ_VERSION_MINOR}_${ZeroMQ_VERSION_PATCH}"
      "libzmq-${CMAKE_VS_PLATFORM_TOOLSET}-mt-${ZeroMQ_VERSION_MAJOR}_${ZeroMQ_VERSION_MINOR}_${ZeroMQ_VERSION_PATCH}"
      libzmq_d
      "libzmq-mt-gd-${ZeroMQ_VERSION_MAJOR}_${ZeroMQ_VERSION_MINOR}_${ZeroMQ_VERSION_PATCH}"
      "libzmq-${CMAKE_VS_PLATFORM_TOOLSET}-mt-gd-${ZeroMQ_VERSION_MAJOR}_${ZeroMQ_VERSION_MINOR}_${ZeroMQ_VERSION_PATCH}"
    HINTS
    	${_ZeroMQ_ROOT}/lib
    )
endif()
```

6. 最后，包含了标准`FindPackageHandleStandardArgs.cmake`，并调用相应的CMake命令。如果找到所有需要的变量，并且版本匹配，则将`ZeroMQ_FOUND`变量设置为`TRUE`:

```cmake
include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(ZeroMQ
  FOUND_VAR
  	ZeroMQ_FOUND
  REQUIRED_VARS
  ZeroMQ_INCLUDE_DIRS
  ZeroMQ_LIBRARIES
  VERSION_VAR
  ZeroMQ_VERSION
  )
```

**NOTE**:*刚才描述的`FindZeroMQ.cmake`模块已经在 https://github.com/zeromq/azmq/blob/master/config/FindZeroMQ.cmake 上进行了修改。*

### 工作原理

`find-module`通常遵循特定的模式:

1. 检查用户是否为所需的包提供了自定义位置。
2. 使用`find_`家族中的命令搜索所需包的必需组件，即头文件、库、可执行程序等等。我们使用`find_path`查找头文件的完整路径，并使用`find_library`查找库。CMake还提供`find_file`、`find_program`和`find_package`。这些命令的签名如下:

```cmake
find_path(<VAR> NAMES name PATHS paths)
```

1. 如果搜索成功，`<VAR>`将保存搜索结果；如果搜索失败，则会设置为`<VAR>-NOTFOUND`。`NAMES`和`PATHS`分别是CMake应该查找的文件的名称和搜索应该指向的路径。
2. 初步搜索的结果中，可以提取版本号。示例中，ZeroMQ头文件包含库版本，可以使用字符串操作和正则表达式提取库版本信息。
3. 最后，调用`find_package_handle_standard_args`命令。处理`find_package`命令的`REQUIRED`、`QUIET`和版本参数，并设置`ZeroMQ_FOUND`变量。

**NOTE**:*任何CMake命令的完整文档都可以从命令行获得。例如，`cmake --help-command find_file`将输出`find_file`命令的手册页。对于CMake标准模块的手册，可以在CLI使用`--help-module`看到。例如，`cmake --help-module FindPackageHandleStandardArgs`将输出`FindPackageHandleStandardArgs.cmake`的手册页面。*

### 更多信息

总而言之，有四种方式可用于找到依赖包:

1. 使用由包供应商提供CMake文件`<package>Config.cmake` ，`<package>ConfigVersion.cmake`和`<package>Targets.cmake`，通常会在包的标准安装位置查找。
2. 无论是由CMake还是第三方提供的模块，为所需包使用`find-module`。
3. 使用`pkg-config`，如本节的示例所示。
4. 如果这些都不可行，那么编写自己的`find`模块。

这四种可选方案按相关性进行了排序，每种方法也都有其挑战。

目前，并不是所有的包供应商都提供CMake的Find文件，不过正变得越来越普遍。因为导出CMake目标，使得第三方代码很容易使用它所依赖的库和/或程序附加的依赖。

从一开始，`Find-module`就一直是CMake中定位依赖的主流手段。但是，它们中的大多数仍然依赖于设置依赖项使用的变量，比如`Boost_INCLUDE_DIRS`、`PYTHON_INTERPRETER`等等。这种方式很难在第三方发布自己的包时，确保依赖关系被满足。

使用`pkg-config`的方法可以很好地进行适配，因为它已经成为Unix系统的标准。然而，也由于这个原因，它不是一个完全跨平台的方法。此外，如CMake文档所述，在某些情况下，用户可能会意外地覆盖检测包，并导致`pkg-config`提供不正确的信息。

最后的方法是编写自己的查找模块脚本，就像本示例中那样。这是可行的，并且依赖于`FindPackageHandleStandardArgs.cmake`。然而，编写一个全面的查找模块脚本绝非易事；有需要考虑很多可能性，我们在Unix和Windows平台上，为查找ZeroMQ库文件演示了一个例子。

所有软件开发人员都非常清楚这些问题和困难，正如CMake邮件列表上讨论所示: https://cmake.org/pipermail/cmake/2018-May/067556.html 。`pkg-config`在Unix包开发人员中是可以接受的，但是它不能很容易地移植到非Unix平台。CMake配置文件功能强大，但并非所有软件开发人员都熟悉CMake语法。公共包规范项目是统一用于包查找的`pkg-config`和CMake配置文件方法的最新尝试。您可以在项目的网站上找到更多信息: https://mwoehlke.github.io/cps/

在第10章中将讨论，如何使用前面讨论中概述的第一种方法，使第三方应用程序，找到自己的包：为项目提供自己的CMake查找文件。



# CMake 完整使用教程 之五 创建和运行测试



本章的主要内容有：

- 创建一个简单的单元测试
- 使用Catch2库进行单元测试
- 使用Google Test库进行单元测试
- 使用Boost Test进行单元测试
- 使用动态分析来检测内存缺陷
- 预期测试失败
- 使用超时测试运行时间过长的测试
- 并行测试
- 运行测试子集
- 使用测试固件

测试代码是开发工具的核心组件。通过单元测试和集成测试自动化测试，不仅可以帮助开发人员尽早回归功能检测，还可以帮助开发人员参与，并了解项目。它可以帮助新开发人员向项目代码提交修改，并确保预期的功能性。对于验证安装是否保留了代码的功能时，自动化测试必不可少。从一开始对单元、模块或库进行测试，可以使用一种纯函数式的风格，将全局变量和全局状态最小化，可让开发者的具有更模块化、更简单的编程风格。

本章中，我们将演示如何使用流行的测试库和框架，将测试集成到CMake构建结构中，并谨记以下目标：

- 让用户、开发人员和持续集成服务很容易地运行测试集。应该像使用`Unix Makefile`时，键入`make test`一样简单。
- 通过最小化测试时间，高效地运行测试，最大限度地提高运行测试的概率——理想情况下，每次代码修改都该如此。

## 创建一个简单的单元测试

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-04/recipe-01 中找到，包含一个C++的示例。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

CTest是CMake的测试工具，本示例中，我们将使用CTest进行单元测试。为了保持对CMake/CTest的关注，我们的测试代码会尽可能的简单。计划是编写和测试能够对整数求和的代码，示例代码只会对整数进行累加，不处理浮点数。就像年轻的卡尔•弗里德里希•高斯(Carl Friedrich Gauss)，被他的老师测试从1到100求和所有自然数一样，我们将要求代码做同样的事情。为了说明CMake没有对实际测试的语言进行任何限制，我们不仅使用C++可执行文件测试代码，还使用Python脚本和shell脚本作为测试代码。为了简单起见，我们将不使用任何测试库来实现，但是我们将在 后面的示例中介绍C++测试框架。

### 准备工作

代码示例由三个文件组成。实现源文件`sum_integs.cpp`对整数向量进行求和，并返回累加结果：

```c++
#include "sum_integers.hpp"

#include <vector>

int sum_integers(const std::vector<int> integers) {
	auto sum = 0;
	for (auto i : integers) {
		sum += i;
	}
	return sum;
}
```

这个示例是否是优雅的实现并不重要，接口以`sum_integers`的形式导出。接口在`sum_integers.hpp`文件中声明，详情如下:

```c++
#pragma once

#include <vector>

int sum_integers(const std::vector<int> integers);
```

最后，main函数在`main.cpp`中定义，从`argv[]`中收集命令行参数，将它们转换成整数向量，调用`sum_integers`函数，并将结果打印到输出中:

```c++
#include "sum_integers.hpp"

#include <iostream>
#include <string>
#include <vector>

// we assume all arguments are integers and we sum them up
// for simplicity we do not verify the type of arguments
int main(int argc, char *argv[]) {
	std::vector<int> integers;
	for (auto i = 1; i < argc; i++) {
		integers.push_back(std::stoi(argv[i]));
	}
	auto sum = sum_integers(integers);
  
	std::cout << sum << std::endl;
}
```

测试这段代码使用C++实现(`test.cpp`)，Bash shell脚本实现(`test.sh`)和Python脚本实现(`test.py`)，只要实现可以返回一个零或非零值，从而CMake可以解释为成功或失败。

C++例子(`test.cpp`)中，我们通过调用`sum_integers`来验证1 + 2 + 3 + 4 + 5 = 15：

```c++
#include "sum_integers.hpp"

#include <vector>

int main() {
	auto integers = {1, 2, 3, 4, 5};
	
  if (sum_integers(integers) == 15) {
		return 0;
	} else {
		return 1;
	}
}
```

Bash shell脚本调用可执行文件：

```bash
#!/usr/bin/env bash

EXECUTABLE=$1

OUTPUT=$($EXECUTABLE 1 2 3 4)

if [ "$OUTPUT" = "10" ]
then
	exit 0
else
	exit 1
fi
```

此外，Python脚本调用可执行文件(使用`--executable`命令行参数传递)，并使用`--short`命令行参数执行：

```python
import subprocess
import argparse

# test script expects the executable as argument
parser = argparse.ArgumentParser()
parser.add_argument('--executable',
										 help='full path to executable')
parser.add_argument('--short',
										 default=False,
                    action='store_true',
                    help='run a shorter test')
args = parser.parse_args()

def execute_cpp_code(integers):
	result = subprocess.check_output([args.executable] + integers)
	return int(result)

if args.short:
	# we collect [1, 2, ..., 100] as a list of strings
	result = execute_cpp_code([str(i) for i in range(1, 101)])
	assert result == 5050, 'summing up to 100 failed'
else:
	# we collect [1, 2, ..., 1000] as a list of strings
	result = execute_cpp_code([str(i) for i in range(1, 1001)])
	assert result == 500500, 'summing up to 1000 failed'
```

### 具体实施

现在，我们将逐步描述如何为项目设置测试：

1. 对于这个例子，我们需要C++11支持，可用的Python解释器，以及Bash shell:

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-01 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PythonInterp REQUIRED)
find_program(BASH_EXECUTABLE NAMES bash REQUIRED)
```

2. 然后，定义库及主要可执行文件的依赖关系，以及测试可执行文件：

```cmake
# example library
add_library(sum_integers sum_integers.cpp)

# main code
add_executable(sum_up main.cpp)
target_link_libraries(sum_up sum_integers)

# testing binary
add_executable(cpp_test test.cpp)
target_link_libraries(cpp_test sum_integers)
```

3. 最后，打开测试功能并定义四个测试。最后两个测试， 调用相同的Python脚本，先没有任何命令行参数，再使用`--short`：

```cmake
enable_testing()

add_test(
  NAME bash_test
  COMMAND ${BASH_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test.sh $<TARGET_FILE:sum_up>
  )
  
add_test(
  NAME cpp_test
  COMMAND $<TARGET_FILE:cpp_test>
  )
  
add_test(
  NAME python_test_long
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test.py --executable $<TARGET_FILE:sum_up>
  )
  
add_test(
  NAME python_test_short
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test.py --short --executable $<TARGET_FILE:sum_up>
  )
```

4. 现在，我们已经准备好配置和构建代码。先手动进行测试：

```bash
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
$ ./sum_up 1 2 3 4 5

15
```

5. 然后，我们可以用`ctest`运行测试集：

```bash
$ ctest

Test project /home/user/cmake-recipes/chapter-04/recipe-01/cxx-example/build
Start 1: bash_test
1/4 Test #1: bash_test ........................ Passed 0.01 sec
Start 2: cpp_test
2/4 Test #2: cpp_test ......................... Passed 0.00 sec
Start 3: python_test_long
3/4 Test #3: python_test_long ................. Passed 0.06 sec
Start 4: python_test_short
4/4 Test #4: python_test_short ................ Passed 0.05 sec
100% tests passed, 0 tests failed out of 4
Total Test time (real) = 0.12 sec
```

6. 还应该尝试中断实现，以验证测试集是否能捕捉到更改。

### 工作原理

这里的两个关键命令：

- `enable_testing()`，测试这个目录和所有子文件夹(因为我们把它放在主`CMakeLists.txt`)。
- `add_test()`，定义了一个新的测试，并设置测试名称和运行命令。

```cmake
add_test(
  NAME cpp_test
  COMMAND $<TARGET_FILE:cpp_test>
  )
```

上面的例子中，使用了生成器表达式:`$<TARGET_FILE:cpp_test>`。生成器表达式，是在生成**构建系统生成时**的表达式。我们将在第5章第9节中详细地描述生成器表达式。此时，我们可以声明`$<TARGET_FILE:cpp_test>`变量，将使用`cpp_test`可执行目标的完整路径进行替换。

生成器表达式在测试时非常方便，因为不必显式地将可执行程序的位置和名称，可以硬编码到测试中。以一种可移植的方式实现这一点非常麻烦，因为可执行文件和可执行后缀(例如，Windows上是`.exe`后缀)的位置在不同的操作系统、构建类型和生成器之间可能有所不同。使用生成器表达式，我们不必显式地了解位置和名称。

也可以将参数传递给要运行的`test`命令，例如：

```cmake
add_test(
  NAME python_test_short
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test.py --short --executable $<TARGET_FILE:sum_up>
  )
```

这个例子中，我们按顺序运行测试，并展示如何缩短总测试时间并行执行测试(第8节)，执行测试用例的子集(第9节)。这里，可以自定义测试命令，可以以任何编程语言运行测试集。CTest关心的是，通过命令的返回码测试用例是否通过。CTest遵循的标准约定是，返回零意味着成功，非零返回意味着失败。可以返回零或非零的脚本，都可以做测试用例。

既然知道了如何定义和执行测试，那么了解如何诊断测试失败也很重要。为此，我们可以在代码中引入一个bug，让所有测试都失败:

```bash
Start 1: bash_test
1/4 Test #1: bash_test ........................***Failed 0.01 sec
	Start 2: cpp_test
2/4 Test #2: cpp_test .........................***Failed 0.00 sec
	Start 3: python_test_long
3/4 Test #3: python_test_long .................***Failed 0.06 sec
	Start 4: python_test_short
4/4 Test #4: python_test_short ................***Failed 0.06 sec

0% tests passed, 4 tests failed out of 4

Total Test time (real) = 0.13 sec

The following tests FAILED:
1 - bash_test (Failed)
2 - cpp_test (Failed)
3 - python_test_long (Failed)
4 - python_test_short (Failed)
Errors while running CTest
```

如果我们想了解更多，可以查看文件`test/Temporary/lasttestsfailure.log`。这个文件包含测试命令的完整输出，并且在分析阶段，要查看的第一个地方。使用以下CLI开关，可以从CTest获得更详细的测试输出：

- `--output-on-failure`:将测试程序生成的任何内容打印到屏幕上，以免测试失败。
- `-v`:将启用测试的详细输出。
- `-vv`:启用更详细的输出。

CTest提供了一个非常方快捷的方式，可以重新运行以前失败的测试；要使用的CLI开关是`--rerun-failed`，在调试期间非常有用。

### 更多信息

考虑以下定义:

```cmake
add_test(
  NAME python_test_long
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test.py --executable $<TARGET_FILE:sum_up>
  )
```

前面的定义可以通过显式指定脚本运行的`WORKING_DIRECTORY`重新表达，如下:

```cmake
add_test(
  NAME python_test_long
  COMMAND ${PYTHON_EXECUTABLE} test.py --executable $<TARGET_FILE:sum_up>
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
```

测试名称可以包含`/`字符，按名称组织相关测试也很有用，例如：

```cmake
add_test(
  NAME python/long
  COMMAND ${PYTHON_EXECUTABLE} test.py --executable $<TARGET_FILE:sum_up>
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
```

有时候，我们需要为测试脚本设置环境变量。这可以通过`set_tests_properties`实现:

```cmake
set_tests_properties(python_test
  PROPERTIES
    ENVIRONMENT
      ACCOUNT_MODULE_PATH=${CMAKE_CURRENT_SOURCE_DIR}
      ACCOUNT_HEADER_FILE=${CMAKE_CURRENT_SOURCE_DIR}/account/account.h
      ACCOUNT_LIBRARY_FILE=$<TARGET_FILE:account>
  )
```

这种方法在不同的平台上并不总可行，CMake提供了解决这个问题的方法。下面的代码片段与上面给出的代码片段相同，在执行实际的Python测试脚本之前，通过`CMAKE_COMMAND`调用CMake来预先设置环境变量:

```cmake
add_test(
  NAME
  	python_test
  COMMAND
    ${CMAKE_COMMAND} -E env
    ACCOUNT_MODULE_PATH=${CMAKE_CURRENT_SOURCE_DIR}
    ACCOUNT_HEADER_FILE=${CMAKE_CURRENT_SOURCE_DIR}/account/account.h
    ACCOUNT_LIBRARY_FILE=$<TARGET_FILE:account>
    ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/account/test.py
  )
```

同样，要注意使用生成器表达式`$<TARGET_FILE:account>`来传递库文件的位置。

我们已经使用`ctest`命令执行测试，CMake还将为生成器创建目标(Unix Makefile生成器为`make test`，Ninja工具为`ninja test`，或者Visual Studio为`RUN_TESTS`)。这意味着，还有另一种(几乎)可移植的方法来运行测试：

```bash
$ cmake --build . --target test
```

不幸的是，当使用Visual Studio生成器时，我们需要使用`RUN_TESTS`来代替:

```bash
$ cmake --build . --target RUN_TESTS
```

**NOTE**:*`ctest`提供了丰富的命令行参数。其中一些内容将在以后的示例中探讨。要获得完整的列表，需要使用`ctest --help`来查看。命令`cmake --help-manual ctest`会将向屏幕输出完整的ctest手册。*

## 使用Catch2库进行单元测试

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-04/recipe-02 中找到，包含一个C++的示例。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

前面的配置中，使用返回码来表示`test.cpp`测试的成功或失败。对于简单功能没问题，但是通常情况下，我们想要使用一个测试框架，它提供了相关基础设施来运行更复杂的测试，包括固定方式进行测试，与数值公差的比较，以及在测试失败时输出更好的错误报告。这里，我们用目前比较流行的测试库Catch2( https://github.com/catchorg/Catch2 )来进行演示。这个测试框架有个很好的特性，它可以通过单个头库包含在项目中进行测试，这使得编译和更新框架特别容易。这个配置中，我们将CMake和Catch2结合使用，来测试上一个求和代码。

我们需要`catch.hpp`头文件，可以从 https://github.com/catchorg/Catch2 (我们使用的是版本2.0.1)下载，并将它与`test.cpp`一起放在项目的根目录下。

### 准备工作

`main.cpp`、`sum_integers.cpp`和`sum_integers.hpp`与之前的示例相同，但将更新`test.cpp`:

```c++
#include "sum_integers.hpp"

// this tells catch to provide a main()
// only do this in one cpp file
#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include <vector>

TEST_CASE("Sum of integers for a short vector", "[short]")
{
  auto integers = {1, 2, 3, 4, 5};
  REQUIRE(sum_integers(integers) == 15);
}

TEST_CASE("Sum of integers for a longer vector", "[long]")
{
  std::vector<int> integers;
  for (int i = 1; i < 1001; ++i)
  {
    integers.push_back(i);
  }
  REQUIRE(sum_integers(integers) == 500500);
}
```

`catch.hpp`头文件可以从https://github.com/catchorg/Catch2 (版本为2.0.1)下载，并将它与`test.cpp`放在项目的根目录中。

### 具体实施

使用Catch2库，需要修改之前的所使用`CMakeList.txt`：

1. 保持`CMakeLists.txt`大多数部分内容不变:

```cmake
# set minimum cmake version
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# project name and language
project(recipe-02 LANGUAGES CXX)

# require C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# example library
add_library(sum_integers sum_integers.cpp)

# main code
add_executable(sum_up main.cpp)
target_link_libraries(sum_up sum_integers)

# testing binary
add_executable(cpp_test test.cpp)
target_link_libraries(cpp_test sum_integers)
```

2. 对于上一个示例的配置，需要保留一个测试，并重命名它。注意，`--success`选项可传递给单元测试的可执行文件。这是一个Catch2选项，测试成功时，也会有输出:

```cmake
enable_testing()

add_test(
  NAME catch_test
  COMMAND $<TARGET_FILE:cpp_test> --success
  )
```

3. 就是这样！让我们来配置、构建和测试。CTest中，使用`-V`选项运行测试，以获得单元测试可执行文件的输出:

```bash
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
$ ctest -V

UpdateCTestConfiguration from :/home/user/cmake-cookbook/chapter-04/recipe-02/cxx-example/build/DartConfiguration.tcl
UpdateCTestConfiguration from :/home/user/cmake-cookbook/chapter-04/recipe-02/cxx-example/build/DartConfiguration.tcl
Test project /home/user/cmake-cookbook/chapter-04/recipe-02/cxx-example/build
Constructing a list of tests
Done constructing a list of tests
Updating test list for fixtures
Added 0 tests to meet fixture requirements
Checking test dependency graph...
Checking test dependency graph end
test 1
Start 1: catch_test
1: Test command: /home/user/cmake-cookbook/chapter-04/recipe-02/cxx-example/build/cpp_test "--success"
1: Test timeout computed to be: 10000000
1:
1: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1: cpp_test is a Catch v2.0.1 host application.
1: Run with -? for options
1:
1: ----------------------------------------------------------------
1: Sum of integers for a short vector
1: ----------------------------------------------------------------
1: /home/user/cmake-cookbook/chapter-04/recipe-02/cxx-example/test.cpp:10
1: ...................................................................
1:
1: /home/user/cmake-cookbook/chapter-04/recipe-02/cxx-example/test.cpp:12:
1: PASSED:
1: REQUIRE( sum_integers(integers) == 15 )
1: with expansion:
1: 15 == 15
1:
1: ----------------------------------------------------------------
1: Sum of integers for a longer vector
1: ----------------------------------------------------------------
1: /home/user/cmake-cookbook/chapter-04/recipe-02/cxx-example/test.cpp:15
1: ...................................................................
1:
1: /home/user/cmake-cookbook/chapter-04/recipe-02/cxx-example/test.cpp:20:
1: PASSED:
1: REQUIRE( sum_integers(integers) == 500500 )
1: with expansion:
1: 500500 (0x7a314) == 500500 (0x7a314)
1:
1: ===================================================================
1: All tests passed (2 assertions in 2 test cases)
1:
1/1 Test #1: catch_test ....................... Passed 0.00 s

100% tests passed, 0 tests failed out of 1

Total Test time (real) = 0.00 se
```

4. 我们也可以测试`cpp_test`的二进制文件，可以直接从Catch2中看到输出:

```bash
$ ./cpp_test --success

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cpp_test is a Catch v2.0.1 host application.
Run with -? for options
-------------------------------------------------------------------
Sum of integers for a short vector
-------------------------------------------------------------------
/home/user/cmake-cookbook/chapter-04/recipe-02/cxx-example/test.cpp:10
...................................................................
/home/user/cmake-cookbook/chapter-04/recipe-02/cxx-example/test.cpp:12:
PASSED:
REQUIRE( sum_integers(integers) == 15 )
with expansion:
15 == 15
-------------------------------------------------------------------
Sum of integers for a longer vector
-------------------------------------------------------------------
/home/user/cmake-cookbook/chapter-04/recipe-02/cxx-example/test.cpp:15
...................................................................
/home/user/cmake-cookbook/chapter-04/recipe-02/cxx-example/test.cpp:20:
PASSED:
REQUIRE( sum_integers(integers) == 500500 )
with expansion:
500500 (0x7a314) == 500500 (0x7a314)
===================================================================
All tests passed (2 assertions in 2 test cases)
```

5. Catch2将生成一个可执行文件，还可以尝试执行以下命令，以探索单元测试框架提供的选项:

```bash
$ ./cpp_test --help
```

### 工作原理

Catch2是一个单头文件测试框架，所以不需要定义和构建额外的目标。只需要确保CMake能找到`catch.hpp`，从而构建`test.cpp`即可。为了方便起见，将它放在与`test.cpp`相同的目录中，我们可以选择一个不同的位置，并使用`target_include_directory`指示该位置。另一种方法是将头部封装到接口库中，这可以在Catch2文档中说明( https://github.com/catchorg/catch2/blob/maste/docs/build.systems.md#cmake ):

```cmake
# Prepare "Catch" library for other executables 
set(CATCH_INCLUDE_DIR
${CMAKE_CURRENT_SOURCE_DIR}/catch) 

add_library(Catch
INTERFACE) 

target_include_directories(Catch INTERFACE
${CATCH_INCLUDE_DIR})
```

然后，我们对库进行如下链接:

```cmake
target_link_libraries(cpp_test Catch)
```

回想一下第3中的讨论，在第1章从简单的可执行库到接口库，是CMake提供的伪目标库，这些伪目标库对于指定项目外部目标的需求非常有用。

### 更多信息

这是一个简单的例子，主要关注CMake。当然，Catch2提供了更多功能。有关Catch2框架的完整文档，可访问 https://github.com/catchorg/Catch2 。

Catch2代码库包含有CMake函数，用于解析Catch测试并自动创建CMake测试，不需要显式地输入`add_test()`函数，可见 https://github.com/catchorg/Catch2/blob/master/contrib/ParseAndAddCatchTests.cmake 。

## 使用Google Test库进行单元测试

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-04/recipe-03 中找到，包含一个C++的示例。该示例在CMake 3.11版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。在代码库中，有一个支持CMake 3.5的例子。*

本示例中，我们将演示如何在CMake的帮助下使用Google Test框架实现单元测试。与前一个配置相比，Google Test框架不仅仅是一个头文件，也是一个库，包含两个需要构建和链接的文件。可以将它们与我们的代码项目放在一起，但是为了使代码项目更加轻量级，我们将选择在配置时，下载一个定义良好的Google Test，然后构建框架并链接它。我们将使用较新的`FetchContent`模块(从CMake版本3.11开始可用)。第8章中会继续讨论`FetchContent`，在这里将讨论模块在底层是如何工作的，并且还将演示如何使用`ExternalProject_Add`进行模拟。此示例的灵感来自(改编自) https://cmake.org/cmake/help/v3.11/module/FetchContent.html 示例。

### 准备工作

`main.cpp`、`sum_integers.cpp`和`sum_integers.hpp`与之前相同，修改`test.cpp`:

```c++
#include "sum_integers.hpp"
#include "gtest/gtest.h"

#include <vector>

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

TEST(example, sum_zero) {
  auto integers = {1, -1, 2, -2, 3, -3};
  auto result = sum_integers(integers);
  ASSERT_EQ(result, 0);
}

TEST(example, sum_five) {
  auto integers = {1, 2, 3, 4, 5};
  auto result = sum_integers(integers);
  ASSERT_EQ(result, 15);
}
```

如上面的代码所示，我们显式地将`gtest.h`，而不将其他Google Test源放在代码项目存储库中，会在配置时使用`FetchContent`模块下载它们。

### 具体实施

下面的步骤描述了如何设置`CMakeLists.txt`，使用GTest编译可执行文件及其相应的测试:

1. 与前两个示例相比，`CMakeLists.txt`的开头基本没有变化，CMake 3.11才能使用`FetchContent`模块:

```cmake
# set minimum cmake version
cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

# project name and language
project(recipe-03 LANGUAGES CXX)

# require C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# example library
add_library(sum_integers sum_integers.cpp)

# main code
add_executable(sum_up main.cpp)
target_link_libraries(sum_up sum_integers)
```

2. 然后引入一个`if`，检查`ENABLE_UNIT_TESTS`。默认情况下，它为`ON`，但有时需要设置为`OFF`，以免在没有网络连接时，也能使用Google Test:

```cmake
option(ENABLE_UNIT_TESTS "Enable unit tests" ON)
message(STATUS "Enable testing: ${ENABLE_UNIT_TESTS}")

if(ENABLE_UNIT_TESTS)
	# all the remaining CMake code will be placed here
endif()
```

3. `if`内部包含`FetchContent`模块，声明要获取的新内容，并查询其属性:

```cmake
include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.8.0
)

FetchContent_GetProperties(googletest)
```

4. 如果内容还没有获取到，将尝试获取并配置它。这需要添加几个可以链接的目标。本例中，我们对`gtest_main`感兴趣。该示例还包含一些变通方法，用于使用在Visual Studio下的编译:

```cmake
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  
  # Prevent GoogleTest from overriding our compiler/linker options
  # when building with Visual Studio
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  # Prevent GoogleTest from using PThreads
  set(gtest_disable_pthreads ON CACHE BOOL "" FORCE)
  
  # adds the targers: gtest, gtest_main, gmock, gmock_main
  add_subdirectory(
    ${googletest_SOURCE_DIR}
    ${googletest_BINARY_DIR}
    )
    
  # Silence std::tr1 warning on MSVC
  if(MSVC)
    foreach(_tgt gtest gtest_main gmock gmock_main)
      target_compile_definitions(${_tgt}
        PRIVATE
        	"_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING"
      )
    endforeach()
  endif()
endif()
```

5. 然后，使用`target_sources`和`target_link_libraries`命令，定义`cpp_test`可执行目标并指定它的源文件:

```cmake
add_executable(cpp_test "")

target_sources(cpp_test
  PRIVATE
  	test.cpp
  )

target_link_libraries(cpp_test
  PRIVATE
    sum_integers
    gtest_main
  )
```

6. 最后，使用`enable_test`和`add_test`命令来定义单元测试:

```cmake
enable_testing()

add_test(
  NAME google_test
  COMMAND $<TARGET_FILE:cpp_test>
  )
```

7. 现在，准备配置、构建和测试项目:

```bash
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
$ ctest

Test project /home/user/cmake-cookbook/chapter-04/recipe-03/cxx-example/build
	Start 1: google_test
1/1 Test #1: google_test ...................... Passed 0.00 sec

100% tests passed, 0 tests failed out of 1

Total Test time (real) = 0.00 sec
```

8. 可以直接运行`cpp_test`:

```bash
$ ./cpp_test

[==========] Running 2 tests from 1 test case.
[----------] Global test environment set-up.
[----------] 2 tests from example
[ RUN ] example.sum_zero
[ OK ] example.sum_zero (0 ms)
[ RUN ] example.sum_five
[ OK ] example.sum_five (0 ms)
[----------] 2 tests from example (0 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 1 test case ran. (0 ms total)
[ PASSED ] 2 tests.
```

### 工作原理

`FetchContent`模块支持通过`ExternalProject`模块，在配置时填充内容，并在其3.11版本中成为CMake的标准部分。而`ExternalProject_Add()`在构建时(见第8章)进行下载操作，这样`FetchContent`模块使得构建可以立即进行，这样获取的主要项目和外部项目(在本例中为Google Test)仅在第一次执行CMake时调用，使用`add_subdirectory`可以嵌套。

为了获取Google Test，首先声明外部内容:

```cmake
include(FetchContent)

FetchContent_Declare(
	googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.8.0
)
```

本例中，我们获取了一个带有特定标记的Git库(release-1.8.0)，但是我们也可以从Subversion、Mercurial或HTTP(S)源获取一个外部项目。有关可用选项，可参考相应的`ExternalProject_Add`命令的选项，网址是https://cmake.org/cmake/help/v3.11/module/ExternalProject.html 。

调用`FetchContent_Populate()`之前，检查是否已经使用`FetchContent_GetProperties()`命令处理了内容填充；否则，调用`FetchContent_Populate()`超过一次后，就会抛出错误。

`FetchContent_Populate(googletest)`用于填充源并定义`googletest_SOURCE_DIR`和`googletest_BINARY_DIR`，可以使用它们来处理Google Test项目(使用`add_subdirectory()`，因为它恰好也是一个CMake项目):

```cmake
add_subdirectory(
  ${googletest_SOURCE_DIR}
  ${googletest_BINARY_DIR}
  )
```

前面定义了以下目标：`gtest`、`gtest_main`、`gmock`和`gmock_main`。这个配置中，作为单元测试示例的库依赖项，我们只对`gtest_main`目标感兴趣：

```cmake
target_link_libraries(cpp_test
  PRIVATE
    sum_integers
    gtest_main
)
```

构建代码时，可以看到如何正确地对Google Test进行配置和构建。有时，我们希望升级到更新的Google Test版本，这时需要更改的唯一一行就是详细说明`GIT_TAG`的那一行。

### 更多信息

了解了`FetchContent`及其构建时的近亲`ExternalProject_Add`，我们将在第8章中重新讨论这些命令。有关可用选项的详细讨论，可参考https://cmake.org/cmake/help/v3.11/module/FetchContent.html 。

本示例中，我们在配置时获取源代码，也可以将它们安装在系统环境中，并使用`FindGTest`模块来检测库和头文件(https://cmake.org/cmake/help/v3.5/module/FindTest.html )。从3.9版开始，CMake还提供了一个Google Test模块(https://cmake.org/cmake/help/v3.9/module/GoogleTest.html )，它提供了一个`gtest_add_tests`函数。通过搜索Google Test宏的源代码，可以使用此函数自动添加测试。

当然，Google Test有许多有趣的的特性，可在 https://github.com/google/googletest 查看。

## 使用Boost Test进行单元测试

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-04/recipe-04 中找到，包含一个C++的示例。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

Boost Test是在C++社区中，一个非常流行的单元测试框架。本例中，我们将演示如何使用Boost Test，对求和示例代码进行单元测试。

### 准备工作

`main.cpp`、`sum_integers.cpp`和`sum_integers.hpp`与之前的示例相同，将更新`test.cpp`作为使用Boost Test库进行的单元测试：



```c++
#include "sum_integers.hpp"

#include <vector>

#define BOOST_TEST_MODULE example_test_suite
#include <boost/test/unit_test.hpp>
BOOST_AUTO_TEST_CASE(add_example)
{
  auto integers = {1, 2, 3, 4, 5};
  auto result = sum_integers(integers);
  BOOST_REQUIRE(result == 15);
}
```

### 具体实施

以下是使用Boost Test构建项目的步骤:

1. 先从`CMakeLists.txt`开始:

```cmake
# set minimum cmake version
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# project name and language
project(recipe-04 LANGUAGES CXX)

# require C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# example library
add_library(sum_integers sum_integers.cpp)

# main code
add_executable(sum_up main.cpp)
target_link_libraries(sum_up sum_integers)
```

2. 检测Boost库并将`cpp_test`链接到它:

```cmake
find_package(Boost 1.54 REQUIRED COMPONENTS unit_test_framework)

add_executable(cpp_test test.cpp)

target_link_libraries(cpp_test
  PRIVATE
    sum_integers
    Boost::unit_test_framework
  )
  
# avoid undefined reference to "main" in test.cpp
target_compile_definitions(cpp_test
  PRIVATE
  	BOOST_TEST_DYN_LINK
  )
```

3. 最后，定义单元测试:

```cmake
enable_testing()

add_test(
  NAME boost_test
  COMMAND $<TARGET_FILE:cpp_test>
  )
```

4. 下面是需要配置、构建和测试代码的所有内容:

```bash
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
$ ctest

Test project /home/user/cmake-recipes/chapter-04/recipe-04/cxx-example/build
Start 1: boost_test
1/1 Test #1: boost_test ....................... Passed 0.01 sec
100% tests passed, 0 tests failed out of 1
Total Test time (real) = 0.01 sec

$ ./cpp_test

Running 1 test case...
*** No errors detected
```

### 工作原理

使用`find_package`来检测Boost的`unit_test_framework`组件(参见第3章，第8节)。我们认为这个组件是`REQUIRED`的，如果在系统环境中找不到它，配置将停止。`cpp_test`目标需要知道在哪里可以找到Boost头文件，并且需要链接到相应的库；它们都由`IMPORTED`库目标`Boost::unit_test_framework`提供，该目标由`find_package`设置。

### 更多信息

本示例中，我们假设系统上安装了Boost。或者，我们可以在编译时获取并构建Boost依赖项。然而，Boost不是轻量级依赖项。我们的示例代码中，我们只使用了最基本的设施，但是Boost提供了丰富的特性和选项，有感兴趣的读者可以去这里看看：http://www.boost.org/doc/libs/1_65_1/libs/test/doc/html/index.html 。

## 使用动态分析来检测内存缺陷

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-04/recipe-05 中找到，包含一个C++的示例。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

内存缺陷：写入或读取越界，或者内存泄漏(已分配但从未释放的内存)，会产生难以跟踪的bug，最好尽早将它们检查出来。Valgrind( [http://valgrind.org](http://valgrind.org/) )是一个通用的工具，用来检测内存缺陷和内存泄漏。本节中，我们将在使用CMake/CTest测试时使用Valgrind对内存问题进行警告。

### 备工作

对于这个配置，需要三个文件。第一个是测试的实现(我们可以调用文件`leaky_implementation.cpp`):

```c++
#include "leaky_implementation.hpp"

int do_some_work() {
  
  // we allocate an array
  double *my_array = new double[1000];
  
  // do some work
  // ...
  
  // we forget to deallocate it
  // delete[] my_array;
  
  return 0;
}
```

还需要相应的头文件(`leaky_implementation.hpp`):

```c++
#pragma once

int do_some_work();
```

并且，需要测试文件(`test.cpp`):

```c++
#include "leaky_implementation.hpp"

int main() {
  int return_code = do_some_work();
  
  return return_code;
}
```

我们希望测试通过，因为`return_code`硬编码为`0`。这里我们也期望检测到内存泄漏，因为`my_array`没有释放。

### 具体实施

下面展示了如何设置CMakeLists.txt来执行代码动态分析:

1. 我们首先定义CMake最低版本、项目名称、语言、目标和依赖关系:

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-05 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(example_library leaky_implementation.cpp)

add_executable(cpp_test test.cpp)
target_link_libraries(cpp_test example_library)
```

2. 然后，定义测试目标，还定义了`MEMORYCHECK_COMMAND`:

```cmake
find_program(MEMORYCHECK_COMMAND NAMES valgrind)
set(MEMORYCHECK_COMMAND_OPTIONS "--trace-children=yes --leak-check=full")

# add memcheck test action
include(CTest)

enable_testing()

add_test(
  NAME cpp_test
  COMMAND $<TARGET_FILE:cpp_test>
  )
```

3. 运行测试集，报告测试通过情况，如下所示:

```bash
$ ctest

Test project /home/user/cmake-recipes/chapter-04/recipe-05/cxx-example/build
Start 1: cpp_test
1/1 Test #1: cpp_test ......................... Passed 0.00 sec
100% tests passed, 0 tests failed out of 1
Total Test time (real) = 0.00 sec
```

4. 现在，我们希望检查内存缺陷，可以观察到被检测到的内存泄漏:

```bash
$ ctest -T memcheck

Site: myhost
Build name: Linux-c++
Create new tag: 20171127-1717 - Experimental
Memory check project /home/user/cmake-recipes/chapter-04/recipe-05/cxx-example/build
Start 1: cpp_test
1/1 MemCheck #1: cpp_test ......................... Passed 0.40 sec
100% tests passed, 0 tests failed out of 1
Total Test time (real) = 0.40 sec
-- Processing memory checking output:
1/1 MemCheck: #1: cpp_test ......................... Defects: 1
MemCheck log files can be found here: ( * corresponds to test number)
/home/user/cmake-recipes/chapter-04/recipe-05/cxx-example/build/Testing/Temporary/MemoryChecker.*.log
Memory checking results:
Memory Leak - 1
```

1. 最后一步，应该尝试修复内存泄漏，并验证`ctest -T memcheck`没有报告错误。

### 工作原理

使用`find_program(MEMORYCHECK_COMMAND NAMES valgrind)`查找valgrind，并将`MEMORYCHECK_COMMAND`设置为其绝对路径。我们显式地包含CTest模块来启用`memcheck`测试操作，可以使用`CTest -T memcheck`来启用这个操作。此外，使用`set(MEMORYCHECK_COMMAND_OPTIONS "--trace-children=yes --leak-check=full")`，将相关参数传递给Valgrind。内存检查会创建一个日志文件，该文件可用于详细记录内存缺陷信息。

**NOTE**:*一些工具，如代码覆盖率和静态分析工具，可以进行类似地设置。然而，其中一些工具的使用更加复杂，因为需要专门的构建和工具链。Sanitizers就是这样一个例子。有关更多信息，请参见https://github.com/arsenm/sanitizers-cmake 。另外，请参阅第14章，其中讨论了AddressSanitizer和ThreadSanitizer。*

### 更多信息

该方法可向测试面板报告内存缺陷，这里演示的功能也可以独立于测试面板使用。我们将在第14章中重新讨论，与CDash一起使用的情况。

有关Valgrind及其特性和选项的文档，请参见[http://valgrind.org](http://valgrind.org/) 。

##  预期测试失败

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-04/recipe-06 中找到，包含一个C++的示例。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

理想情况下，我们希望所有的测试能在每个平台上通过。然而，也可能想要测试预期的失败或异常是否会在受控的设置中进行。这种情况下，我们将把预期的失败定义为成功。我们认为，这通常应该交给测试框架(例如：Catch2或Google Test)的任务，它应该检查预期的失败并向CMake报告成功。但是，在某些情况下，您可能希望将测试的非零返回代码定义为成功；换句话说，您可能想要颠倒成功和失败的定义。在本示例中，我们将演示这种情况。

### 准备工作

这个配置的测试用例是一个很小的Python脚本(`test.py`)，它总是返回1，CMake将其解释为失败:

```python
import sys

# simulate a failing test
sys.exit(1)
```

### 实施步骤

如何编写CMakeLists.txt来完成我们的任务:

1. 这个示例中，不需要任何语言支持从CMake，但需要Python:

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(recipe-06 LANGUAGES NONE)
find_package(PythonInterp REQUIRED)
```

2. 然后，定义测试并告诉CMake，测试预期会失败:

```cmake
enable_testing()
add_test(example ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test.py)
set_tests_properties(example PROPERTIES WILL_FAIL true)
```

3. 最后，报告是一个成功的测试，如下所示:

```bash
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
$ ctest

Test project /home/user/cmake-recipes/chapter-04/recipe-06/example/build
Start 1: example
1/1 Test #1: example .......................... Passed 0.00 sec
100% tests passed, 0 tests failed out of 1
Total Test time (real) = 0.01 sec
```

### 工作原理

使用`set_tests_properties(example PROPERTIES WILL_FAIL true)`，将属性`WILL_FAIL`设置为`true`，这将转换成功与失败。但是，这个特性不应该用来临时修复损坏的测试。

### 更多信息

如果需要更大的灵活性，可以将测试属性`PASS_REGULAR_EXPRESSION`和`FAIL_REGULAR_EXPRESSION`与`set_tests_properties`组合使用。如果设置了这些参数，测试输出将根据参数给出的正则表达式列表进行检查，如果匹配了正则表达式，测试将通过或失败。可以在测试中设置其他属性，完整的属性列表可以参考：https://cmake.org/cmake/help/v3.5/manual/cmake-properties.7.html#properties-on-tests 。

## 使用超时测试运行时间过长的测试

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-04/recipe-07 中找到。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

理想情况下，测试集应该花很短的时间进行，以便开发人员经常运行测试，并使每个提交(变更集)进行测试成为可能(或更容易)。然而，有些测试可能会花费更长的时间或者被卡住(例如，由于高文件I/O负载)，我们可能需要设置超时来终止耗时过长的测试，它们延迟了整个测试，并阻塞了部署管道。本示例中，我们将演示一种设置超时的方法，可以针对每个测试设置不同的超时。

### 准备工作

这个示例是一个Python脚本(`test.py`)，它总是返回0。为了保持这种简单性，并保持对CMake方面的关注，测试脚本除了等待两秒钟外什么也不做。实际中，这个测试脚本将执行更有意义的工作:

```python
import sys
import time

# wait for 2 seconds
time.sleep(2)

# report success
sys.exit(0)
```

### 具体实施

我们需要通知CTest终止测试，如下:

1. 我们定义项目名称，启用测试，并定义测试:

```cmake
# set minimum cmake version
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# project name
project(recipe-07 LANGUAGES NONE)

# detect python
find_package(PythonInterp REQUIRED)

# define tests
enable_testing()

# we expect this test to run for 2 seconds
add_test(example ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test.py)
```

2. 另外，我们为测试指定时限，设置为10秒:

```cmake
set_tests_properties(example PROPERTIES TIMEOUT 10)
```

3. 知道了如何进行配置和构建，并希望测试能够通过:

```bash
$ ctest

Test project /home/user/cmake-recipes/chapter-04/recipe-07/example/build
Start 1: example
1/1 Test #1: example .......................... Passed 2.01 sec
100% tests passed, 0 tests failed out of 1
Total Test time (real) = 2.01 sec
```

4. 现在，为了验证超时是否有效，我们将`test.py`中的`sleep`命令增加到11秒，并重新运行测试:

```bash
$ ctest

Test project /home/user/cmake-recipes/chapter-04/recipe-07/example/build
Start 1: example
1/1 Test #1: example ..........................***Timeout 10.01 sec
0% tests passed, 1 tests failed out of 1
Total Test time (real) = 10.01 sec
The following tests FAILED:
1 - example (Timeout)
Errors while running CTest
```

### 工作原理

`TIMEOUT`是一个方便的属性，可以使用`set_tests_properties`为单个测试指定超时时间。如果测试运行超过了这个设置时间，不管出于什么原因(测试已经停止或者机器太慢)，测试将被终止并标记为失败。

## 并行测试

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-04/recipe-08 中找到。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

大多数现代计算机都有4个或更多个CPU核芯。CTest有个非常棒的特性，能够并行运行测试，如果您有多个可用的核。这可以减少测试的总时间，而减少总测试时间才是真正重要的，从而开发人员频繁地进行测试。本示例中，我们将演示这个特性，并讨论如何优化测试以获得最大的性能。

其他测试可以进行相应地表示，我们把这些测试脚本放在`CMakeLists.txt`同目录下面的test目录中。

### 准备工作

我们假设测试集包含标记为a, b，…，j的测试用例，每一个都有特定的持续时间:

| 测试用例   | 该单元的耗时 |
| :--------- | :----------- |
| a, b, c, d | 0.5          |
| e, f, g    | 1.5          |
| h          | 2.5          |
| i          | 3.5          |
| j          | 4.5          |

时间单位可以是分钟，但是为了保持简单和简短，我们将使用秒。为简单起见，我们可以用Python脚本表示`test a`，它消耗0.5个时间单位:

```python
import sys
import time

# wait for 0.5 seconds
time.sleep(0.5)

# finally report success
sys.exit(0)
```

其他测试同理。我们将把这些脚本放在`CMakeLists.txt`下面，一个名为`test`的目录中。

### 具体实施

对于这个示例，我们需要声明一个测试列表，如下:

1. `CMakeLists.txt`非常简单：

```cmake
# set minimum cmake version
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# project name
project(recipe-08 LANGUAGES NONE)

# detect python
find_package(PythonInterp REQUIRED)

# define tests
enable_testing()
add_test(a ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/a.py)
add_test(b ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/b.py)
add_test(c ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/c.py)
add_test(d ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/d.py)
add_test(e ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/e.py)
add_test(f ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/f.py)
add_test(g ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/g.py)
add_test(h ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/h.py)
add_test(i ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/i.py)
add_test(j ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/j.py)
```

2. 我们可以配置项目，使用`ctest`运行测试，总共需要17秒:

```bash
$ mkdir -p build
$ cd build
$ cmake ..
$ ctest

Start 1: a
1/10 Test #1: a ................................ Passed 0.51 sec
Start 2: b
2/10 Test #2: b ................................ Passed 0.51 sec
Start 3: c
3/10 Test #3: c ................................ Passed 0.51 sec
Start 4: d
4/10 Test #4: d ................................ Passed 0.51 sec
Start 5: e
5/10 Test #5: e ................................ Passed 1.51 sec
Start 6: f
6/10 Test #6: f ................................ Passed 1.51 sec
Start 7: g
7/10 Test #7: g ................................ Passed 1.51 sec
Start 8: h
8/10 Test #8: h ................................ Passed 2.51 sec
Start 9: i
9/10 Test #9: i ................................ Passed 3.51 sec
Start 10: j
10/10 Test #10: j ................................ Passed 4.51 sec
100% tests passed, 0 tests failed out of 10
Total Test time (real) = 17.11 sec
```

3. 现在，如果机器有4个内核可用，我们可以在不到5秒的时间内在4个内核上运行测试集:

```bash
$ ctest --parallel 4

Start 10: j
Start 9: i
Start 8: h
Start 5: e
1/10 Test #5: e ................................ Passed 1.51 sec
Start 7: g
2/10 Test #8: h ................................ Passed 2.51 sec
Start 6: f
3/10 Test #7: g ................................ Passed 1.51 sec
Start 3: c
4/10 Test #9: i ................................ Passed 3.63 sec
5/10 Test #3: c ................................ Passed 0.60 sec
Start 2: b
Start 4: d
6/10 Test #6: f ................................ Passed 1.51 sec
7/10 Test #4: d ................................ Passed 0.59 sec
8/10 Test #2: b ................................ Passed 0.59 sec
Start 1: a
9/10 Test #10: j ................................ Passed 4.51 sec
10/10 Test #1: a ................................ Passed 0.51 sec
100% tests passed, 0 tests failed out of 10
Total Test time (real) = 4.74 sec
```

### 工作原理

可以观察到，在并行情况下，测试j、i、h和e同时开始。当并行运行时，总测试时间会有显著的减少。观察`ctest --parallel 4`的输出，我们可以看到并行测试运行从最长的测试开始，最后运行最短的测试。从最长的测试开始是一个非常好的策略。这就像打包移动的盒子：从较大的项目开始，然后用较小的项目填补空白。a-j测试在4个核上的叠加比较，从最长的开始，如下图所示:

```bash
--> time
core 1: jjjjjjjjj
core 2: iiiiiiibd
core 3: hhhhhggg
core 4: eeefffac
```

按照定义测试的顺序运行，运行结果如下:

```bash
--> time
core 1: aeeeiiiiiii
core 2: bfffjjjjjjjjj
core 3: cggg
core 4: dhhhhh
```

按照定义测试的顺序运行测试，总的来说需要更多的时间，因为这会让2个核大部分时间处于空闲状态(这里的核3和核4)。CMake知道每个测试的时间成本，是因为我们先顺序运行了测试，将每个测试的成本数据记录在`test/Temporary/CTestCostData.txt`文件中:

```bash
a 1 0.506776
b 1 0.507882
c 1 0.508175
d 1 0.504618
e 1 1.51006
f 1 1.50975
g 1 1.50648
h 1 2.51032
i 1 3.50475
j 1 4.51111
```

如果在配置项目之后立即开始并行测试，它将按照定义测试的顺序运行测试，在4个核上的总测试时间明显会更长。这意味着什么呢？这意味着，我们应该减少的时间成本来安排测试？这是一种决策，但事实证明还有另一种方法，我们可以自己表示每次测试的时间成本:

```cmake
add_test(a ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/a.py)
add_test(b ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/b.py)
add_test(c ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/c.py)
add_test(d ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/d.py)
set_tests_properties(a b c d PROPERTIES COST 0.5)

add_test(e ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/e.py)
add_test(f ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/f.py)
add_test(g ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/g.py)
set_tests_properties(e f g PROPERTIES COST 1.5)

add_test(h ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/h.py)
set_tests_properties(h PROPERTIES COST 2.5)

add_test(i ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/i.py)
set_tests_properties(i PROPERTIES COST 3.5)

add_test(j ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/j.py)
set_tests_properties(j PROPERTIES COST 4.5)
```

成本参数可以是一个估计值，也可以从`test/Temporary/CTestCostData.txt`中提取。

### 更多信息

除了使用`ctest --parallel N`，还可以使用环境变量`CTEST_PARALLEL_LEVEL`将其设置为所需的级别。

## 运行测试子集

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-04/recipe-09 中找到。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

前面的示例中，我们学习了如何在CMake的帮助下并行运行测试，并讨论了从最长的测试开始是最高效的。虽然，这种策略将总测试时间最小化，但是在特定特性的代码开发期间，或者在调试期间，我们可能不希望运行整个测试集。对于调试和代码开发，我们只需要能够运行选定的测试子集。在本示例中，我们将实现这一策略。

### 准备工作

在这个例子中，我们假设总共有六个测试：前三个测试比较短，名称分别为`feature-a`、`feature-b`和`feature-c`，还有三个长测试，名称分别是`feature-d`、`benchmark-a`和`benchmark-b`。这个示例中，我们可以用Python脚本表示这些测试，可以在其中调整休眠时间:

```python
import sys
import time

# wait for 0.1 seconds
time.sleep(0.1)

# finally report success
sys.exit(0)
```

### 具体实施

以下是我们CMakeLists.txt文件内容的详细内容:

1. `CMakeLists.txt`中，定义了六个测试:

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# project name
project(recipe-09 LANGUAGES NONE)

# detect python
find_package(PythonInterp REQUIRED)

# define tests
enable_testing()

add_test(
  NAME feature-a
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/feature-a.py
  )
add_test(
  NAME feature-b
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/feature-b.py
  )
add_test(
  NAME feature-c
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/feature-c.py
  )
add_test(
  NAME feature-d
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/feature-d.py
  )
add_test(
  NAME benchmark-a
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/benchmark-a.py
  )
add_test(
  NAME benchmark-b
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/benchmark-b.py
  )
```

2. 此外，我们给较短的测试贴上`quick`的标签，给较长的测试贴上`long`的标签:

```cmake
set_tests_properties(
  feature-a
  feature-b
  feature-c
  PROPERTIES
  	LABELS "quick"
  )
set_tests_properties(
  feature-d
  benchmark-a
  benchmark-b
  PROPERTIES
  	LABELS "long"
  )
```

3. 我们现在可以运行测试集了，如下:

```bash
$ mkdir -p build
$ cd build
$ cmake ..
$ ctest

Start 1: feature-a
1/6 Test #1: feature-a ........................ Passed 0.11 sec
Start 2: feature-b
2/6 Test #2: feature-b ........................ Passed 0.11 sec
Start 3: feature-c
3/6 Test #3: feature-c ........................ Passed 0.11 sec
Start 4: feature-d
4/6 Test #4: feature-d ........................ Passed 0.51 sec
Start 5: benchmark-a
5/6 Test #5: benchmark-a ...................... Passed 0.51 sec
Start 6: benchmark-b
6/6 Test #6: benchmark-b ...................... Passed 0.51 sec
100% tests passed, 0 tests failed out of 6
Label Time Summary:
long = 1.54 sec*proc (3 tests)
quick = 0.33 sec*proc (3 tests)
Total Test time (real) = 1.87 sec
```

### 工作原理

现在每个测试都有一个名称和一个标签。CMake中所有的测试都是有编号的，所以它们也带有唯一编号。定义了测试标签之后，我们现在可以运行整个集合，或者根据它们的名称(使用正则表达式)、标签或编号运行测试。

按名称运行测试(运行所有具有名称匹配功能的测试):

```bash
$ ctest -R feature

Start 1: feature-a
1/4 Test #1: feature-a ........................ Passed 0.11 sec
Start 2: feature-b
2/4 Test #2: feature-b ........................ Passed 0.11 sec
Start 3: feature-c
3/4 Test #3: feature-c ........................ Passed 0.11 sec
Start 4: feature-d
4/4 Test #4: feature-d ........................ Passed 0.51 sec
100% tests passed, 0 tests failed out of 4
```

按照标签运行测试(运行所有的长测试):

```bash
$ ctest -L long

Start 4: feature-d
1/3 Test #4: feature-d ........................ Passed 0.51 sec
Start 5: benchmark-a
2/3 Test #5: benchmark-a ...................... Passed 0.51 sec
Start 6: benchmark-b
3/3 Test #6: benchmark-b ...................... Passed 0.51 sec
100% tests passed, 0 tests failed out of 3
```

根据数量运行测试(运行测试2到4)产生的结果是:

```bash
$ ctest -I 2,4

Start 2: feature-b
1/3 Test #2: feature-b ........................ Passed 0.11 sec
Start 3: feature-c
2/3 Test #3: feature-c ........................ Passed 0.11 sec
Start 4: feature-d
3/3 Test #4: feature-d ........................ Passed 0.51 sec
100% tests passed, 0 tests failed out of 3
```

### 更多信息

尝试使用`$ ctest --help`，将看到有大量的选项可供用来定制测试。

## 使用测试固件

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-04/recipe-10 中找到。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

这个示例的灵感来自于Craig Scott，我们建议读者也参考相应的博客文章来了解更多的背景知识，https://crascit.com/2016/10/18/test-fixtures-withcmake-ctest/ ，此示例的动机是演示如何使用测试固件。这对于更复杂的测试非常有用，这些测试需要在测试运行前进行设置，以及在测试完成后执行清理操作(例如：创建示例数据库、设置连接、断开连接、清理测试数据库等等)。我们需要运行一个设置或清理操作的测试，并能够以一种可预测和健壮的方式自动触发这些步骤，而不需要引入代码重复。这些设置和清理步骤可以委托给测试框架(例如Google Test或Catch2)，我们在这里将演示如何在CMake级别实现测试固件。

### 准备工作

我们将准备4个Python脚本，并将它们放在`test`目录下:`setup.py`、`features-a.py`、`features-b.py`和`clean-up.py`。

### 具体实施

我们从`CMakeLists.txt`结构开始，附加一些步骤如下:

1. 基础CMake语句:

```cmake
# set minimum cmake version
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# project name
project(recipe-10 LANGUAGES NONE)

# detect python
find_package(PythonInterp REQUIRED)

# define tests
enable_testing()
```

2. 然后，定义了4个测试步骤，并将它们绑定到一个固件上:

```cmake
add_test(
  NAME setup
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/setup.py
  )
set_tests_properties(
  setup
  PROPERTIES
  	FIXTURES_SETUP my-fixture
  )
add_test(
  NAME feature-a
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/feature-a.py
  )
add_test(
  NAME feature-b
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/feature-b.py
  )
set_tests_properties(
  feature-a
  feature-b
  PROPERTIES
  	FIXTURES_REQUIRED my-fixture
  )
add_test(
  NAME cleanup
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/cleanup.py
  )
set_tests_properties(
  cleanup
  PROPERTIES
  	FIXTURES_CLEANUP my-fixture
  )
```

3. 运行整个集合，如下面的输出所示:

```bash
$ mkdir -p build
$ cd build
$ cmake ..
$ ctest

Start 1: setup
1/4 Test #1: setup ............................ Passed 0.01 sec
Start 2: feature-a
2/4 Test #2: feature-a ........................ Passed 0.01 sec
Start 3: feature-b
3/4 Test #3: feature-b ........................ Passed 0.00 sec
Start 4: cleanup
4/4 Test #4: cleanup .......................... Passed 0.01 sec

100% tests passed, 0 tests failed out of 4
```

4. 然而，当我们试图单独运行测试特性时。它正确地调用设置步骤和清理步骤:

```bash
$ ctest -R feature-a

Start 1: setup
1/3 Test #1: setup ............................ Passed 0.01 sec
Start 2: feature-a
2/3 Test #2: feature-a ........................ Passed 0.00 sec
Start 4: cleanup
3/3 Test #4: cleanup .......................... Passed 0.01 sec

100% tests passed, 0 tests failed out of 3
```

### 工作原理

在本例中，我们定义了一个文本固件，并将其称为`my-fixture`。我们为安装测试提供了`FIXTURES_SETUP`属性，并为清理测试了`FIXTURES_CLEANUP`属性，并且使用`FIXTURES_REQUIRED`，我们确保测试`feature-a`和`feature-b`都需要安装和清理步骤才能运行。将它们绑定在一起，可以确保在定义良好的状态下，进入和离开相应的步骤。



# CMake 完整使用教程 之六 配置时和构建时的操作



本章的主要内容有：

- 使用平台无关的文件操作
- 配置时运行自定义命令
- 构建时运行自定义命令:Ⅰ. 使用add_custom_command
- 构建时运行自定义命令:Ⅱ. 使用add_custom_target
- 构建时为特定目标运行自定义命令
- 探究编译和链接命令
- 探究编译器标志命令
- 探究可执行命令
- 使用生成器表达式微调配置和编译

我们将学习如何在配置和构建时，执行自定义操作。先简单回顾一下，与CMake工作流程相关的时序:

1. **CMake时**或**构建时**：CMake正在运行，并处理项目中的`CMakeLists.txt`文件。
2. **生成时**：生成构建工具(如Makefile或Visual Studio项目文件)。
3. **构建时**：由CMake生成相应平台的原生构建脚本，在脚本中调用原生工具构建。此时，将调用编译器在特定的构建目录中构建目标(可执行文件和库)。
4. **CTest时**或**测试时**：运行测试套件以检查目标是否按预期执行。
5. **CDash时**或**报告时**：当测试结果上传到仪表板上，与其他开发人员共享测试报告。
6. **安装时**：当目标、源文件、可执行程序和库，从构建目录安装到相应位置。
7. **CPack时**或**打包时**：将项目打包用以分发时，可以是源码，也可以是二进制。
8. **包安装时**：新生成的包在系统范围内安装。

本章会介绍在配置和构建时的自定义行为，我们将学习如何使用这些命令:

- **execute_process**，从CMake中执行任意进程，并检索它们的输出。
- **add_custom_target**，创建执行自定义命令的目标。
- **add_custom_command**，指定必须执行的命令，以生成文件或在其他目标的特定生成事件中生成。

## 使用平台无关的文件操作

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-5/recipe-01 中找到，其中包含一个C++例子。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

有些项目构建时，可能需要与平台的文件系统进行交互。也就是检查文件是否存在、创建新文件来存储临时信息、创建或提取打包文件等等。使用CMake不仅能够在不同的平台上生成构建系统，还能够在不复杂的逻辑情况下，进行文件操作，从而独立于操作系统。本示例将展示，如何以可移植的方式下载库文件。

### 准备工作

我们将展示如何提取Eigen库文件，并使用提取的源文件编译我们的项目。这个示例中，将重用第3章第7节的线性代数例子`linear-algebra.cpp`，用来检测外部库和程序、检测特征库。这里，假设已经包含Eigen库文件，已在项目构建前下载。

### 具体实施

项目需要解压缩Eigen打包文件，并相应地为目标设置包含目录:

1. 首先，使能C++11项目:

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-01 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

2. 我们将自定义目标添加到构建系统中，自定义目标将提取构建目录中的库文件:

```cmake
add_custom_target(unpack-eigen
  ALL
  COMMAND
  	${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_SOURCE_DIR}/eigen-eigen-5a0156e40feb.tar.gz
  COMMAND
  	${CMAKE_COMMAND} -E rename eigen-eigen-5a0156e40feb eigen-3.3.4
  WORKING_DIRECTORY
  	${CMAKE_CURRENT_BINARY_DIR}
  COMMENT
  	"Unpacking Eigen3 in ${CMAKE_CURRENT_BINARY_DIR}/eigen-3.3.4"
  )
```

3. 为源文件添加了一个可执行目标:

```cmake
add_executable(linear-algebra linear-algebra.cpp)
```

4. 由于源文件的编译依赖于Eigen头文件，需要显式地指定可执行目标对自定义目标的依赖关系:

```cmake
add_dependencies(linear-algebra unpack-eigen)
```

5. 最后，指定包含哪些目录:

```cmake
target_include_directories(linear-algebra
  PRIVATE
  	${CMAKE_CURRENT_BINARY_DIR}/eigen-3.3.4
  )
```

### 工作原理

细看`add_custom_target`这个命令：

```cmake
add_custom_target(unpack-eigen
  ALL
  COMMAND
  	${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_SOURCE_DIR}/eigen-eigen-5a0156e40feb.tar.gz
  COMMAND
  	${CMAKE_COMMAND} -E rename eigen-eigen-5a0156e40feb eigen-3.3.4
  WORKING_DIRECTORY
  	${CMAKE_CURRENT_BINARY_DIR}
  COMMENT
  	"Unpacking Eigen3 in ${CMAKE_CURRENT_BINARY_DIR}/eigen-3.3.4"
  )
```

构建系统中引入了一个名为`unpack-eigen`的目标。因为我们传递了`ALL`参数，目标将始终被执行。`COMMAND`参数指定要执行哪些命令。本例中，我们希望提取存档并将提取的目录重命名为`egan -3.3.4`，通过以下两个命令实现:

```cmake
${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_SOURCE_DIR}/eigen-eigen-
5a0156e40feb.tar.gz
${CMAKE_COMMAND} -E rename eigen-eigen-5a0156e40feb eigen-3.3.4
```

注意，使用`-E`标志调用CMake命令本身来执行实际的工作。对于许多常见操作，CMake实现了一个对所有操作系统都通用的接口，这使得构建系统独立于特定的平台。`add_custom_target`命令中的下一个参数是工作目录。我们的示例中，它对应于构建目录：`CMAKE_CURRENT_BINARY_DIR`。最后一个参数`COMMENT`，用于指定CMake在执行自定义目标时输出什么样的消息。

### 更多信息

构建过程中必须执行一系列没有输出的命令时，可以使用`add_custom_target`命令。正如我们在本示例中所示，可以将自定义目标指定为项目中其他目标的依赖项。此外，自定义目标还可以依赖于其他目标。

使用`-E`标志可以以与操作系统无关的方式，运行许多公共操作。运行`cmake -E`或`cmake -E help`可以获得特定操作系统的完整列表。例如，这是Linux系统上命令的摘要:

```bash
Usage: cmake -E <command> [arguments...]
Available commands:
  capabilities              - Report capabilities built into cmake in JSON format
  chdir dir cmd [args...]   - run command in a given directory
  compare_files file1 file2 - check if file1 is same as file2
  copy <file>... destination  - copy files to destination (either file or directory)
  copy_directory <dir>... destination   - copy content of <dir>... directories to 'destination' directory
  copy_if_different <file>... destination  - copy files if it has changed
  echo [<string>...]        - displays arguments as text
  echo_append [<string>...] - displays arguments as text but no new line
  env [--unset=NAME]... [NAME=VALUE]... COMMAND [ARG]...
                            - run command in a modified environment
  environment               - display the current environment
  make_directory <dir>...   - create parent and <dir> directories
  md5sum <file>...          - create MD5 checksum of files
  sha1sum <file>...         - create SHA1 checksum of files
  sha224sum <file>...       - create SHA224 checksum of files
  sha256sum <file>...       - create SHA256 checksum of files
  sha384sum <file>...       - create SHA384 checksum of files
  sha512sum <file>...       - create SHA512 checksum of files
  remove [-f] <file>...     - remove the file(s), use -f to force it
  remove_directory dir      - remove a directory and its contents
  rename oldname newname    - rename a file or directory (on one volume)
  server                    - start cmake in server mode
  sleep <number>...         - sleep for given number of seconds
  tar [cxt][vf][zjJ] file.tar [file/dir1 file/dir2 ...]
                            - create or extract a tar or zip archive
  time command [args...]    - run command and display elapsed time
  touch file                - touch a file.
  touch_nocreate file       - touch a file but do not create it.
Available on UNIX only:
  create_symlink old new    - create a symbolic link new -> old
```

## 配置时运行自定义命令

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-05/recipe-02 中找到。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

运行CMake生成构建系统，从而指定原生构建工具必须执行哪些命令，以及按照什么顺序执行。我们已经了解了CMake如何在配置时运行许多子任务，以便找到工作的编译器和必要的依赖项。本示例中，我们将讨论如何使用`execute_process`命令在配置时运行定制化命令。

### 具体实施

第3章第3节中，我们已经展示了`execute_process`查找Python模块NumPy时的用法。本例中，我们将使用`execute_process`命令来确定，是否存在特定的Python模块(本例中为Python CFFI)，如果存在，我们在进行版本确定:

1. 对于这个简单的例子，不需要语言支持:

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(recipe-02 LANGUAGES NONE)
```

2. 我们要求Python解释器执行一个简短的代码片段，因此，需要使用`find_package`来查找解释器：

```cmake
find_package(PythonInterp REQUIRED)
```

3. 然后，调用`execute_process`来运行一个简短的Python代码段；下一节中，我们将更详细地讨论这个命令:

```cmake
# this is set as variable to prepare
# for abstraction using loops or functions
set(_module_name "cffi")

execute_process(
  COMMAND
  	${PYTHON_EXECUTABLE} "-c" "import ${_module_name}; print(${_module_name}.__version__)"
  OUTPUT_VARIABLE _stdout
  ERROR_VARIABLE _stderr
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_STRIP_TRAILING_WHITESPACE
  )
```

4. 然后，打印结果：

```cmake
if(_stderr MATCHES "ModuleNotFoundError")
	message(STATUS "Module ${_module_name} not found")
else()
	message(STATUS "Found module ${_module_name} v${_stdout}")
endif()
```

5. 下面是一个配置示例(假设Python CFFI包安装在相应的Python环境中):

```bash
$ mkdir -p build
$ cd build
$ cmake ..

-- Found PythonInterp: /home/user/cmake-cookbook/chapter-05/recipe-02/example/venv/bin/python (found version "3.6.5")
-- Found module cffi v1.11.5
```

### 工作原理

`execute_process`命令将从当前正在执行的CMake进程中派生一个或多个子进程，从而提供了在配置项目时运行任意命令的方法。可以在一次调用`execute_process`时执行多个命令。但请注意，每个命令的输出将通过管道传输到下一个命令中。该命令接受多个参数:

- WORKING_DIRECTORY，指定应该在哪个目录中执行命令。
- RESULT_VARIABLE将包含进程运行的结果。这要么是一个整数，表示执行成功，要么是一个带有错误条件的字符串。
- OUTPUT_VARIABLE和ERROR_VARIABLE将包含执行命令的标准输出和标准错误。由于命令的输出是通过管道传输的，因此只有最后一个命令的标准输出才会保存到OUTPUT_VARIABLE中。
- INPUT_FILE指定标准输入重定向的文件名
- OUTPUT_FILE指定标准输出重定向的文件名
- ERROR_FILE指定标准错误输出重定向的文件名
- 设置OUTPUT_QUIET和ERROR_QUIET后，CMake将静默地忽略标准输出和标准错误。
- 设置OUTPUT_STRIP_TRAILING_WHITESPACE，可以删除运行命令的标准输出中的任何尾随空格
- 设置ERROR_STRIP_TRAILING_WHITESPACE，可以删除运行命令的错误输出中的任何尾随空格。

有了这些了解这些参数，回到我们的例子当中:

```cmake
set(_module_name "cffi")

execute_process(
  COMMAND
  	${PYTHON_EXECUTABLE} "-c" "import ${_module_name}; print(${_module_name}.__version__)"
  OUTPUT_VARIABLE _stdout
  ERROR_VARIABLE _stderr
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_STRIP_TRAILING_WHITESPACE
  )
if(_stderr MATCHES "ModuleNotFoundError")
	message(STATUS "Module ${_module_name} not found")
else()
  message(STATUS "Found module ${_module_name} v${_stdout}")
endif()
```

该命令检查`python -c "import cffi; print(cffi.__version__)"`的输出。如果没有找到模块，`_stderr`将包含`ModuleNotFoundError`，我们将在if语句中对其进行检查。本例中，我们将打印`Module cffi not found`。如果导入成功，Python代码将打印模块的版本，该模块通过管道输入`_stdout`，这样就可以打印如下内容:

```cmake
message(STATUS "Found module ${_module_name} v${_stdout}")
```

### 更多信息

本例中，只打印了结果，但实际项目中，可以警告、中止配置，或者设置可以查询的变量，来切换某些配置选项。

代码示例会扩展到多个Python模块(如Cython)，以避免代码重复。一种选择是使用`foreach`循环模块名，另一种方法是将代码封装为函数或宏。我们将在第7章中讨论这些封装。

第9章中，我们将使用Python CFFI和Cython。现在的示例，可以作为有用的、可重用的代码片段，来检测这些包是否存在。

## 构建时运行自定义命令:Ⅰ. 使用add_custom_command

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-5/recipe-03 中找到，其中包含一个C++例子。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

项目的构建目标取决于命令的结果，这些命令只能在构建系统生成完成后的构建执行。CMake提供了三个选项来在构建时执行自定义命令:

1. 使用`add_custom_command`编译目标，生成输出文件。
2. `add_custom_target`的执行没有输出。
3. 构建目标前后，`add_custom_command`的执行可以没有输出。

这三个选项强制执行特定的语义，并且不可互换。接下来的三个示例将演示具体的用法。

### 准备工作

我们将重用第3章第4节中的C++示例，以说明如何使用`add_custom_command`的第一个选项。代码示例中，我们了解了现有的BLAS和LAPACK库，并编译了一个很小的C++包装器库，以调用线性代数的Fortran实现。

我们将把代码分成两部分。`linear-algebra.cpp`的源文件与第3章、第4章没有区别，并且将包含线性代数包装器库的头文件和针对编译库的链接。源代码将打包到一个压缩的tar存档文件中，该存档文件随示例项目一起提供。存档文件将在构建时提取，并在可执行文件生成之前，编译线性代数的包装器库。

### 具体实施

`CMakeLists.txt`必须包含一个自定义命令，来提取线性代数包装器库的源代码：

1. 从CMake最低版本、项目名称和支持语言的定义开始:

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(recipe-03 LANGUAGES CXX Fortran)
```

2. 选择C++11标准:

```cmake
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

3. 然后，在系统上查找BLAS和LAPACK库:

```cmake
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
```

4. 声明一个变量`wrap_BLAS_LAPACK_sources`来保存`wrap_BLAS_LAPACK.tar.gz`压缩包文件的名称:

```cmake
set(wrap_BLAS_LAPACK_sources
  ${CMAKE_CURRENT_BINARY_DIR}/wrap_BLAS_LAPACK/CxxBLAS.hpp
  ${CMAKE_CURRENT_BINARY_DIR}/wrap_BLAS_LAPACK/CxxBLAS.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/wrap_BLAS_LAPACK/CxxLAPACK.hpp
  ${CMAKE_CURRENT_BINARY_DIR}/wrap_BLAS_LAPACK/CxxLAPACK.cpp
  )
```

5. 声明自定义命令来提取`wrap_BLAS_LAPACK.tar.gz`压缩包，并更新提取文件的时间戳。注意这个`wrap_BLAS_LAPACK_sources`变量的预期输出:

```cmake
add_custom_command(
  OUTPUT
  	${wrap_BLAS_LAPACK_sources}
  COMMAND
  	${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_SOURCE_DIR}/wrap_BLAS_LAPACK.tar.gz
  COMMAND
  	${CMAKE_COMMAND} -E touch ${wrap_BLAS_LAPACK_sources}
  WORKING_DIRECTORY
  	${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
  	${CMAKE_CURRENT_SOURCE_DIR}/wrap_BLAS_LAPACK.tar.gz
  COMMENT
  	"Unpacking C++ wrappers for BLAS/LAPACK"
  VERBATIM
  )
```

6. 接下来，添加一个库目标，源文件是新解压出来的:

```cmake
add_library(math "")

target_sources(math
  PRIVATE
  	${CMAKE_CURRENT_BINARY_DIR}/wrap_BLAS_LAPACK/CxxBLAS.cpp
  	${CMAKE_CURRENT_BINARY_DIR}/wrap_BLAS_LAPACK/CxxLAPACK.cpp
  PUBLIC
  	${CMAKE_CURRENT_BINARY_DIR}/wrap_BLAS_LAPACK/CxxBLAS.hpp
  	${CMAKE_CURRENT_BINARY_DIR}/wrap_BLAS_LAPACK/CxxLAPACK.hpp
  )
  
target_include_directories(math
  INTERFACE
  	${CMAKE_CURRENT_BINARY_DIR}/wrap_BLAS_LAPACK
  )
  
target_link_libraries(math
  PUBLIC
  	${LAPACK_LIBRARIES}
  )
```

7. 最后，添加`linear-algebra`可执行目标。可执行目标链接到库:

```cmake
add_executable(linear-algebra linear-algebra.cpp)

target_link_libraries(linear-algebra
  PRIVATE
  	math
  )
```

8. 我们配置、构建和执行示例:

```bash
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
$ ./linear-algebra 1000

C_DSCAL done
C_DGESV done
info is 0
check is 4.35597e-10
```

### 工作原理

让我们来了解一下`add_custom_command`的使用:

```cmake
add_custom_command(
  OUTPUT
  	${wrap_BLAS_LAPACK_sources}
  COMMAND
  	${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_SOURCE_DIR}/wrap_BLAS_LAPACK.tar.gz
  COMMAND
  	${CMAKE_COMMAND} -E touch ${wrap_BLAS_LAPACK_sources}
  WORKING_DIRECTORY
  	${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
  	${CMAKE_CURRENT_SOURCE_DIR}/wrap_BLAS_LAPACK.tar.gz
  COMMENT
  	"Unpacking C++ wrappers for BLAS/LAPACK"
  VERBATIM
  )
```

`add_custom_command`向目标添加规则，并通过执行命令生成输出。`add_custom_command`中声明的任何目标，即在相同的`CMakeLists.txt`中声明的任何目标，使用输出的任何文件作为源文件的目标，在构建时会有规则生成这些文件。因此，源文件生成在构建时，目标和自定义命令在构建系统生成时，将自动处理依赖关系。

我们的例子中，输出是压缩`tar`包，其中包含有源文件。要检测和使用这些文件，必须在构建时提取打包文件。通过使用带有`-E`标志的CMake命令，以实现平台独立性。下一个命令会更新提取文件的时间戳。这样做是为了确保没有处理陈旧文件。`WORKING_DIRECTORY`可以指定在何处执行命令。示例中，`CMAKE_CURRENT_BINARY_DIR`是当前正在处理的构建目录。`DEPENDS`参数列出了自定义命令的依赖项。例子中，压缩的`tar`是一个依赖项。CMake使用`COMMENT`字段在构建时打印状态消息。最后，`VERBATIM`告诉CMake为生成器和平台生成正确的命令，从而确保完全独立。

我们来仔细看看这用使用方式和打包库的创建：

```cmake
add_library(math "")

target_sources(math
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/wrap_BLAS_LAPACK/CxxBLAS.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/wrap_BLAS_LAPACK/CxxLAPACK.cpp
  PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}/wrap_BLAS_LAPACK/CxxBLAS.hpp
    ${CMAKE_CURRENT_BINARY_DIR}/wrap_BLAS_LAPACK/CxxLAPACK.hpp
  )
  
target_include_directories(math
  INTERFACE
  	${CMAKE_CURRENT_BINARY_DIR}/wrap_BLAS_LAPACK
  )
  
target_link_libraries(math
  PUBLIC
  	${LAPACK_LIBRARIES}
  )
```

我们声明一个没有源的库目标，是因为后续使用`target_sources`填充目标的源。这里实现了一个非常重要的目标，即让依赖于此目标的目标，了解需要哪些目录和头文件，以便成功地使用库。C++源文件的目标是`PRIVATE`，因此只用于构建库。因为目标及其依赖项都需要使用它们来成功编译，所以头文件是`PUBLIC`。包含目录使用`target_include_categories`指定，其中`wrap_BLAS_LAPACK`声明为`INTERFACE`，因为只有依赖于`math`目标的目标需要它。

`add_custom_command`有两个限制:

- 只有在相同的`CMakeLists.txt`中，指定了所有依赖于其输出的目标时才有效。
- 对于不同的独立目标，使用`add_custom_command`的输出可以重新执行定制命令。这可能会导致冲突，应该避免这种情况的发生。

第二个限制，可以使用`add_dependencies`来避免。不过，规避这两个限制的正确方法是使用`add_custom_target`命令，我们将在下一节的示例中详细介绍。

## 构建时运行自定义命令:Ⅱ. 使用add_custom_target

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-5/recipe-04 中找到，其中包含一个C++例子。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

我们在前面的示例，讨论了`add_custom_command`有一些限制，可以通过`add_custom_target`绕过这些限制。这个CMake命令将引入新的目标，与`add_custom_command`相反，这些目标依次执行不返回输出。可以将`add_custom_target`和`add_custom_command`结合使用。使用这种方法，可以与其依赖项所在目录不同的目录指定自定义目标，CMake基础设施对项目设计模块化非常有用。

### 准备工作

我们将重用前一节示例，对源码进行简单的修改。特别是，将把压缩后的`tar`打包文件放在名为`deps`的子目录中，而不是存储在主目录中。这个子目录包含它自己的`CMakeLists.txt`，将由主`CMakeLists.txt`调用。

### 具体实施

我们将从主`CMakeLists.txt`开始，然后讨论`deps/CMakeLists.txt`:

1. 声明启用C++11：

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-04 LANGUAGES CXX Fortran)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

2. 现在，继续讨论`deps/CMakeLists.txt`。这通过`add_subdirectory`命令实现:

```cmake
add_subdirectory(deps)
```

3. `deps/CMakeLists.txt`中，我们首先定位必要的库(BLAS和LAPACK):

```cmake
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
```

4. 然后，我们将`tar`包的内容汇集到一个变量`MATH_SRCS`中:

```cmake
set(MATH_SRCS
  ${CMAKE_CURRENT_BINARY_DIR}/wrap_BLAS_LAPACK/CxxBLAS.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/wrap_BLAS_LAPACK/CxxLAPACK.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/wrap_BLAS_LAPACK/CxxBLAS.hpp
  ${CMAKE_CURRENT_BINARY_DIR}/wrap_BLAS_LAPACK/CxxLAPACK.hpp
  )
```

5. 列出要打包的源之后，定义一个目标和一个命令。这个组合用于提取`${CMAKE_CURRENT_BINARY_DIR}`中的包。但是，这里我们在一个不同的范围内，引用`deps/CMakeLists.txt`，因此`tar`包将存放在到主项目构建目录下的`deps`子目录中:

```cmake
add_custom_target(BLAS_LAPACK_wrappers
  WORKING_DIRECTORY
  	${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
  	${MATH_SRCS}
  COMMENT
  	"Intermediate BLAS_LAPACK_wrappers target"
  VERBATIM
  )

add_custom_command(
  OUTPUT
  	${MATH_SRCS}
  COMMAND
  	${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_SOURCE_DIR}/wrap_BLAS_LAPACK.tar.gz
  WORKING_DIRECTORY
  	${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
  	${CMAKE_CURRENT_SOURCE_DIR}/wrap_BLAS_LAPACK.tar.gz
  COMMENT
  	"Unpacking C++ wrappers for BLAS/LAPACK"
  )
```

6. 添加数学库作为目标，并指定相应的源，包括目录和链接库:

```cmake
add_library(math "")

target_sources(math
  PRIVATE
  	${MATH_SRCS}
  )

target_include_directories(math
  INTERFACE
  	${CMAKE_CURRENT_BINARY_DIR}/wrap_BLAS_LAPACK
  )

# BLAS_LIBRARIES are included in LAPACK_LIBRARIES
target_link_libraries(math
  PUBLIC
  	${LAPACK_LIBRARIES}
  )
```

7. 执行完`deps/CMakeLists.txt`中的命令，返回到父范围，定义可执行目标，并将其链接到另一个目录的数学库:

```cmake
add_executable(linear-algebra linear-algebra.cpp)

target_link_libraries(linear-algebra
  PRIVATE
  	math
  )
```

### 工作原理

用户可以使用`add_custom_target`，在目标中执行定制命令。这与我们前面讨论的`add_custom_command`略有不同。`add_custom_target`添加的目标没有输出，因此总会执行。因此，可以在子目录中引入自定义目标，并且仍然能够在主`CMakeLists.txt`中引用它。

本例中，使用`add_custom_target`和`add_custom_command`提取了源文件的包。这些源文件稍后用于编译另一个库，我们设法在另一个(父)目录范围内链接这个库。构建`CMakeLists.txt`文件的过程中，`tar`包是在`deps`下，`deps`是项目构建目录下的一个子目录。这是因为在CMake中，构建树的结构与源树的层次结构相同。

这个示例中有一个值得注意的细节，就是我们把数学库的源标记为`PRIVATE`:

```cmake
set(MATH_SRCS
  ${CMAKE_CURRENT_BINARY_DIR}/wrap_BLAS_LAPACK/CxxBLAS.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/wrap_BLAS_LAPACK/CxxLAPACK.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/wrap_BLAS_LAPACK/CxxBLAS.hpp
  ${CMAKE_CURRENT_BINARY_DIR}/wrap_BLAS_LAPACK/CxxLAPACK.hpp
  )

# ...

add_library(math "")

target_sources(math
  PRIVATE
  	${MATH_SRCS}
  )

# ...
```

虽然这些源代码是`PRIVATE`，但我们在父范围内编译了`linear-algebra.cpp`，并且这个源代码包括`CxxBLAS.hpp`和`CxxLAPACK.hpp`。为什么这里使用`PRIVATE`，以及如何编译`linear-algebra.cpp`，并构建可执行文件呢？如果将头文件标记为`PUBLIC`, CMake就会在创建时停止，并出现一个错误，“无法找到源文件”，因为要生成(提取)还不存在于文件树中的源文件。

这是一个已知的限制(参见https://gitlab.kitware.com/cmake/cmake/issues/1633 ，以及相关的博客文章:https://samthursfield.wordpress.com/2015/11/21/cmake-depende-ncies-targets-and-files-and-custom-commands )。我们通过声明源代码为`PRIVATE`来解决这个限制。这样CMake时，没有获得对不存在源文件的依赖。但是，CMake内置的C/C++文件依赖关系扫描器在构建时获取它们，并编译和链接源代码。

## 构建时为特定目标运行自定义命令

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-5/recipe-05 中找到，其中包含一个Fortran例子。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

本节示例将展示，如何使用`add_custom_command`的第二个参数，来执行没有输出的自定义操作，这对于构建或链接特定目标之前或之后执行某些操作非常有用。由于自定义命令仅在必须构建目标本身时才执行，因此我们实现了对其执行的目标级控制。我们将通过一个示例来演示，在构建目标之前打印目标的链接，然后在编译后，立即测量编译后，可执行文件的静态分配大小。

### 准备工作

本示例中，我们将使用Fortran代码(`example.f90`):

```fortran
program example

  implicit none
  
  real(8) :: array(20000000)
  real(8) :: r
  integer :: i
  
  do i = 1, size(array)
    call random_number(r)
    array(i) = r
  end do
  
  print *, sum(array)
  
end program
```

虽然我们选择了Fortran，但Fortran代码对于后面的讨论并不重要，因为有很多遗留的Fortran代码，存在静态分配大小的问题。

这段代码中，我们定义了一个包含20,000,000双精度浮点数的数组，这个数组占用160MB的内存。在这里，我们并不是推荐这样的编程实践。一般来说，这些内存的分配和代码中是否使用这段内存无关。一个更好的方法是只在需要时动态分配数组，随后立即释放。

示例代码用随机数填充数组，并计算它们的和——这样是为了确保数组确实被使用，并且编译器不会优化分配。我们将使用Python脚本(`static-size.py`)来统计二进制文件静态分配的大小，该脚本用size命令来封装:

```python
import subprocess
import sys

# for simplicity we do not check number of
# arguments and whether the file really exists
file_path = sys.argv[-1]
try:
	output = subprocess.check_output(['size', file_path]).decode('utf-8')
except FileNotFoundError:
	print('command "size" is not available on this platform')
	sys.exit(0)
  
size = 0.0
for line in output.split('\n'):
	if file_path in line:
		# we are interested in the 4th number on this line
		size = int(line.split()[3])
    
print('{0:.3f} MB'.format(size/1.0e6))
```

要打印链接行，我们将使用第二个Python helper脚本(`echo-file.py`)打印文件的内容:

```python
import sys

# for simplicity we do not verify the number and
# type of arguments
file_path = sys.argv[-1]
try:
	with open(file_path, 'r') as f:
print(f.read())
except FileNotFoundError:
	print('ERROR: file {0} not found'.format(file_path))

```

### 具体实施

来看看`CMakeLists.txt`：

1. 首先声明一个Fortran项目:

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(recipe-05 LANGUAGES Fortran)
```

2. 例子依赖于Python解释器，所以以一种可移植的方式执行helper脚本:

```cmake
find_package(PythonInterp REQUIRED)
```

3. 本例中，默认为“Release”构建类型，以便CMake添加优化标志:

```cmake
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
```

4. 现在，定义可执行目标:

```cmake
add_executable(example "")

target_sources(example
  PRIVATE
  	example.f90
  )
```

5. 然后，定义一个自定义命令，在`example`目标在已链接之前，打印链接行:

```cmake
add_custom_command(
  TARGET
  	example
  PRE_LINK
  	COMMAND
  		${PYTHON_EXECUTABLE}
  		${CMAKE_CURRENT_SOURCE_DIR}/echo-file.py
			${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/example.dir/link.txt
  COMMENT
  	"link line:"
  VERBATIM
  )
```

6. 测试一下。观察打印的链接行和可执行文件的静态大小:

```bash
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .

Scanning dependencies of target example
[ 50%] Building Fortran object CMakeFiles/example.dir/example.f90.o
[100%] Linking Fortran executable example
link line:
/usr/bin/f95 -O3 -DNDEBUG -O3 CMakeFiles/example.dir/example.f90.o -o example
static size of executable:
160.003 MB
[100%] Built target example
```

### 工作原理

当声明了库或可执行目标，就可以使用`add_custom_command`将其他命令锁定到目标上。这些命令将在特定的时间执行，与它们所附加的目标的执行相关联。CMake通过以下选项，定制命令执行顺序:

- **PRE_BUILD**：在执行与目标相关的任何其他规则之前执行的命令。
- **PRE_LINK**：使用此选项，命令在编译目标之后，调用链接器或归档器之前执行。Visual Studio 7或更高版本之外的生成器中使用`PRE_BUILD`将被解释为`PRE_LINK`。
- **POST_BUILD**：如前所述，这些命令将在执行给定目标的所有规则之后运行。

本例中，将两个自定义命令绑定到可执行目标。`PRE_LINK`命令将`${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/example.dir/link.txt`的内容打印到屏幕上。在我们的例子中，链接行是这样的:

```bash
link line:
/usr/bin/f95 -O3 -DNDEBUG -O3 CMakeFiles/example.dir/example.f90.o -o example
```

使用Python包装器来实现这一点，它依赖于shell命令。

第二步中，`POST_BUILD`自定义命令调用Python helper脚本`static-size.py`，生成器表达式`$<target_file:example>`作为参数。CMake将在生成时(即生成生成系统时)将生成器表达式扩展到目标文件路径。然后，Python脚本`static-size.py`使用size命令获取可执行文件的静态分配大小，将其转换为MB，并打印结果。我们的例子中，获得了预期的160 MB:

```bash
static size of executable:
160.003 MB
```

## 探究编译和链接命令



**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-5/recipe-06 中找到，其中包含一个C++例子。该示例在CMake 3.9版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。代码库还有一个与CMake 3.5兼容的示例。*

生成构建系统期间最常见的操作，是试图评估在哪种系统上构建项目。这意味着要找出哪些功能工作，哪些不工作，并相应地调整项目的编译。使用的方法是查询依赖项是否被满足的信号，或者在代码库中是否启用工作区。接下来的几个示例，将展示如何使用CMake执行这些操作。我们将特别讨论以下事宜:

1. 如何确保代码能成功编译为可执行文件。
2. 如何确保编译器理解相应的标志。
3. 如何确保特定代码能成功编译为运行可执行程序。

### 准备工作

示例将展示如何使用来自对应的`Check<LANG>SourceCompiles.cmake`标准模块的`check_<lang>_source_compiles`函数，以评估给定编译器是否可以将预定义的代码编译成可执行文件。该命令可帮助你确定:

- 编译器支持所需的特性。
- 链接器工作正常，并理解特定的标志。
- 可以使用`find_package`找到的包含目录和库。

本示例中，我们将展示如何检测OpenMP 4.5标准的循环特性，以便在C++可执行文件中使用。使用一个C++源文件，来探测编译器是否支持这样的特性。CMake提供了一个附加命令`try_compile`来探究编译。本示例将展示，如何使用这两种方法。

**TIPS**:*可以使用CMake命令行界面来获取关于特定模块(`cmake --help-module <module-name>`)和命令(`cmake --help-command <command-name>`)的文档。示例中，`cmake --help-module CheckCXXSourceCompiles`将把`check_cxx_source_compiles`函数的文档输出到屏幕上，而`cmake --help-command try_compile`将对`try_compile`命令执行相同的操作。*

### 具体实施

我们将同时使用`try_compile`和`check_cxx_source_compiles`，并比较这两个命令的工作方式:

1. 创建一个C++11工程：

```cmake
cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(recipe-06 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

2. 查找比一起支持的OpenMP:

```cmake
find_package(OpenMP)

if(OpenMP_FOUND)
	# ... <- the steps below will be placed here
else()
	message(STATUS "OpenMP not found: no test for taskloop is run")
endif()
```

3. 如果找到OpenMP，再检查所需的特性是否可用。为此，设置了一个临时目录，`try_compile`将在这个目录下来生成中间文件。我们把它放在前面步骤中引入的`if`语句中:

```cmake
set(_scratch_dir ${CMAKE_CURRENT_BINARY_DIR}/omp_try_compile)
```

4. 调用`try_compile`生成一个小项目，以尝试编译源文件`taskloop.cpp`。编译成功或失败的状态，将保存到`omp_taskloop_test_1`变量中。需要为这个示例编译设置适当的编译器标志、包括目录和链接库。因为使用导入的目标`OpenMP::OpenMP_CXX`，所以只需将`LINK_LIBRARIES`选项设置为`try_compile`即可。如果编译成功，则任务循环特性可用，我们为用户打印一条消息:

```cmake
try_compile(
  omp_taskloop_test_1
  	${_scratch_dir}
  SOURCES
  	${CMAKE_CURRENT_SOURCE_DIR}/taskloop.cpp
  LINK_LIBRARIES
  	OpenMP::OpenMP_CXX
  )
message(STATUS "Result of try_compile: ${omp_taskloop_test_1}")

```

5. 要使用`check_cxx_source_compiles`函数，需要包含`CheckCXXSourceCompiles.cmake`模块文件。其他语言也有类似的模块文件，C(`CheckCSourceCompiles.cmake`)和Fortran(`CheckFortranSourceCompiles.cmake`):

```cmake
include(CheckCXXSourceCompiles)
```

6. 我们复制源文件的内容，通过`file(READ ...)`命令读取内容到一个变量中，试图编译和连接这个变量:

```cmake
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/taskloop.cpp _snippet)
```

7. 我们设置了`CMAKE_REQUIRED_LIBRARIES`。这对于下一步正确调用编译器是必需的。注意使用导入的`OpenMP::OpenMP_CXX`目标，它还将设置正确的编译器标志和包含目录:

```cmake
set(CMAKE_REQUIRED_LIBRARIES OpenMP::OpenMP_CXX)
```

8. 使用代码片段作为参数，调用`check_cxx_source_compiles`函数。检查结果将保存到`omp_taskloop_test_2`变量中:

```cmake
check_cxx_source_compiles("${_snippet}" omp_taskloop_test_2)
```

9. 调用`check_cxx_source_compiles`并向用户打印消息之前，我们取消了变量的设置:

```cmake
unset(CMAKE_REQUIRED_LIBRARIES)
message(STATUS "Result of check_cxx_source_compiles: ${omp_taskloop_test_2}"
```

10. 最后，进行测试：

```bash
$ mkdir -p build
$ cd build
$ cmake ..

-- ...
-- Found OpenMP_CXX: -fopenmp (found version "4.5")
-- Found OpenMP: TRUE (found version "4.5")
-- Result of try_compile: TRUE
-- Performing Test omp_taskloop_test_2
-- Performing Test omp_taskloop_test_2 - Success
-- Result of check_cxx_source_compiles: 1
```

### 工作原理

`try_compile`和`check_cxx_source_compiles`都将编译源文件，并将其链接到可执行文件中。如果这些操作成功，那么输出变量`omp_task_loop_test_1`(前者)和`omp_task_loop_test_2`(后者)将被设置为`TRUE`。然而，这两个命令实现的方式略有不同。`check_<lang>_source_compiles`命令是`try_compile`命令的简化包装。因此，它提供了一个接口:

1. 要编译的代码片段必须作为CMake变量传入。大多数情况下，这意味着必须使用`file(READ ...)`来读取文件。然后，代码片段被保存到构建目录的`CMakeFiles/CMakeTmp`子目录中。
2. 微调编译和链接，必须通过设置以下CMake变量进行:
   - CMAKE_REQUIRED_FLAGS：设置编译器标志。
   - CMAKE_REQUIRED_DEFINITIONS：设置预编译宏。
   - CMAKE_REQUIRED_INCLUDES：设置包含目录列表。
   - CMAKE_REQUIRED_LIBRARIES：设置可执行目标能够连接的库列表。
3. 调用`check_<lang>_compiles_function`之后，必须手动取消对这些变量的设置，以确保后续使用中，不会保留当前内容。

**NOTE**:*使用CMake 3.9中可以对于OpenMP目标进行导入,但是目前的配置也可以使用CMake的早期版本，通过手动为`check_cxx_source_compiles`设置所需的标志和库:`set(CMAKE_REQUIRED_FLAGS ${OpenMP_CXX_FLAGS})`和`set(CMAKE_REQUIRED_LIBRARIES ${OpenMP_CXX_LIBRARIES})`。*

**TIPS**:*Fortran下，CMake代码的格式通常是固定的，但也有意外情况。为了处理这些意外，需要为`check_fortran_source_compiles`设置`-ffree-form`编译标志。可以通过`set(CMAKE_REQUIRED_FLAGS “-ffree-form")`实现。*

这个接口反映了：测试编译是通过，在CMake调用中直接生成和执行构建和连接命令来执行的。

命令`try_compile`提供了更完整的接口和两种不同的操作模式:

1. 以一个完整的CMake项目作为输入，并基于它的`CMakeLists.txt`配置、构建和链接。这种操作模式提供了更好的灵活性，因为要编译项目的复杂度是可以选择的。
2. 提供了源文件，和用于包含目录、链接库和编译器标志的配置选项。

因此，`try_compile`基于在项目上调用CMake，其中`CMakeLists.txt`已经存在(在第一种操作模式中)，或者基于传递给`try_compile`的参数动态生成文件。

### 更多信息

本示例中概述的类型检查并不总是万无一失的，并且可能产生假阳性和假阴性。作为一个例子，可以尝试注释掉包含`CMAKE_REQUIRED_LIBRARIES`的行。运行这个例子仍然会报告“成功”，这是因为编译器将忽略OpenMP的`pragma`字段。

当返回了错误的结果时，应该怎么做？构建目录的`CMakeFiles`子目录中的`CMakeOutput.log`和`CMakeError.log`文件会提供一些线索。它们记录了CMake运行的操作的标准输出和标准错误。如果怀疑结果有误，应该通过搜索保存编译检查结果的变量集来检查前者。如果你怀疑有误报，你应该检查后者。

调试`try_compile`需要一些注意事项。即使检查不成功，CMake也会删除由该命令生成的所有文件。幸运的是，`debug-trycompile`将阻止CMake进行删除。如果你的代码中有多个`try_compile`调用，一次只能调试一个:

1. 运行CMake，不使用`--debug-trycompile`，将运行所有`try_compile`命令，并清理它们的执行目录和文件。
2. 从CMake缓存中删除保存检查结果的变量。缓存保存到`CMakeCache.txt`文件中。要清除变量的内容，可以使用`-U`的CLI开关，后面跟着变量的名称，它将被解释为一个全局表达式，因此可以使用`*`和`?`：

```bash
$ cmake -U <variable-name>
```

1. 再次运行CMake，使用`--debug-trycompile`。只有清除缓存的检查才会重新运行。这次不会清理执行目录和文件。

**TIPS**:*`try_compile`提供了灵活和干净的接口，特别是当编译的代码不是一个简短的代码时。我们建议在测试编译时，小代码片段时使用`check_<lang>_source_compile`。其他情况下，选择`try_compile`。*

## 探究编译器标志命令

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-5/recipe-07 中找到，其中包含一个C++例子。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

设置编译器标志，对是否能正确编译至关重要。不同的编译器供应商，为类似的特性实现有不同的标志。即使是来自同一供应商的不同编译器版本，在可用标志上也可能存在细微的差异。有时，会引入一些便于调试或优化目的的新标志。本示例中，我们将展示如何检查所选编译器是否可用某些标志。

### 准备工作

Sanitizers(请参考https://github.com/google/Sanitizers )已经成为静态和动态代码分析的非常有用的工具。通过使用适当的标志重新编译代码并链接到必要的库，可以检查内存错误(地址清理器)、未初始化的读取(内存清理器)、线程安全(线程清理器)和未定义的行为(未定义的行为清理器)相关的问题。与同类型分析工具相比，Sanitizers带来的性能损失通常要小得多，而且往往提供关于检测到的问题的更详细的信息。缺点是，代码(可能还有工具链的一部分)需要使用附加的标志重新编译。

本示例中，我们将设置一个项目，使用不同的Sanitizers来编译代码，并展示如何检查，编译器标志是否正确使用。

### 具体实施

Clang编译器已经提供了Sanitizers，GCC也将其引入工具集中。它们是为C和C++程序而设计的。最新版本的Fortran也能使用这些编译标志，并生成正确的仪表化库和可执行程序。不过，本文将重点介绍C++示例。

1. 声明一个C++11项目：

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(recipe-07 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

2. 声明列表`CXX_BASIC_FLAGS`，其中包含构建项目时始终使用的编译器标志`-g3`和`-O1`:

```cmake
list(APPEND CXX_BASIC_FLAGS "-g3" "-O1")
```

3. 这里需要包括CMake模块`CheckCXXCompilerFlag.cmake`。C的模块为`CheckCCompilerFlag.cmake`，Fotran的模块为`CheckFortranCompilerFlag.cmake`(Fotran的模块是在CMake 3.3添加)

```cmake
include(CheckCXXCompilerFlag)
```

4. 我们声明一个`ASAN_FLAGS`变量，它包含Sanitizer所需的标志，并设置`CMAKE_REQUIRED_FLAGS`变量，`check_cxx_compiler_flag`函数在内部使用该变量:

```cmake
set(ASAN_FLAGS "-fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_REQUIRED_FLAGS ${ASAN_FLAGS})
```

5. 我们调用`check_cxx_compiler_flag`来确保编译器理解`ASAN_FLAGS`变量中的标志。调用函数后，我们取消设置`CMAKE_REQUIRED_FLAGS`:

```cmake
check_cxx_compiler_flag(${ASAN_FLAGS} asan_works)
unset(CMAKE_REQUIRED_FLAGS)
```

6. 如果编译器理解这些选项，我们将变量转换为一个列表，用分号替换空格:

```cmake
if(asan_works)
	string(REPLACE " " ";" _asan_flags ${ASAN_FLAGS})
```

7. 我们添加了一个可执行的目标，为代码定位Sanitizer:

```cmake
add_executable(asan-example asan-example.cpp)
```

8. 我们为可执行文件设置编译器标志，以包含基本的和Sanitizer标志:

```cmake
target_compile_options(asan-example
  PUBLIC
    ${CXX_BASIC_FLAGS}
    ${_asan_flags}
  )
```

9. 最后，我们还将Sanitizer标志添加到链接器使用的标志集中。这将关闭`if(asan_works)`块:

```cmake
target_link_libraries(asan-example PUBLIC ${_asan_flags})
endif()
```

完整的示例源代码还展示了如何编译和链接线程、内存和未定义的行为清理器的示例可执行程序。这里不详细讨论这些，因为我们使用相同的模式来检查编译器标志。

**NOTE**:*在GitHub上可以找到一个定制的CMake模块，用于在您的系统上寻找对Sanitizer的支持:https://github.com/arsenm/sanitizers-cmake*

### 工作原理

`check_<lang>_compiler_flag`函数只是`check_<lang>_source_compiles`函数的包装器。这些包装器为特定代码提供了一种快捷方式。在用例中，检查特定代码片段是否编译并不重要，重要的是编译器是否理解一组标志。

Sanitizer的编译器标志也需要传递给链接器。可以使用`check_<lang>_compiler_flag`函数来实现，我们需要在调用之前设置`CMAKE_REQUIRED_FLAGS`变量。否则，作为第一个参数传递的标志将只对编译器使用。

当前配置中需要注意的是，使用字符串变量和列表来设置编译器标志。使用`target_compile_options`和`target_link_libraries`函数的字符串变量，将导致编译器和/或链接器报错。CMake将传递引用的这些选项，从而导致解析错误。这说明有必要用列表和随后的字符串操作来表示这些选项，并用分号替换字符串变量中的空格。实际上，CMake中的列表是分号分隔的字符串。

### 更多信息

我们将在第7章，编写一个函数来测试和设置编译器标志，到时候再来回顾，并概括测试和设置编译器标志的模式。

## 探究可执行命令

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-5/recipe-08 中找到，其中包含一个C/C++例子。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

目前为止，我们已经展示了如何检查给定的源代码，是否可以由所选的编译器编译，以及如何确保所需的编译器和链接器标志可用。此示例中，将显示如何检查是否可以在当前系统上编译、链接和运行代码。

### 准备工作

本示例的代码示例是复用第3章第9节的配置，并进行微小的改动。之前，我们展示了如何在您的系统上找到ZeroMQ库并将其链接到一个C程序中。本示例中，在生成实际的C++程序之前，我们将检查一个使用GNU/Linux上的系统UUID库的小型C程序是否能够实际运行。

### 具体实施

开始构建C++项目之前，我们希望检查GNU/Linux上的UUID系统库是否可以被链接。这可以通过以下一系列步骤来实现:

1. 声明一个混合的C和C++11程序。这是必要的，因为我们要编译和运行的测试代码片段是使用C语言完成:

```cmake
cmake_minimum_required(VERSION 3.6 FATAL_ERROR)
project(recipe-08 LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

2. 我们需要在系统上找到UUID库。这通过使用`pkg-config`实现的。要求搜索返回一个CMake导入目标使用`IMPORTED_TARGET`参数:

```cmake
find_package(PkgConfig REQUIRED QUIET)
pkg_search_module(UUID REQUIRED uuid IMPORTED_TARGET)
if(TARGET PkgConfig::UUID)
	message(STATUS "Found libuuid")
endif()
```

3. 接下来，需要使用`CheckCSourceRuns.cmake`模块。C++的是`CheckCXXSourceRuns.cmake`模块。但到CMake 3.11为止，Fortran语言还没有这样的模块:

```cmake
include(CheckCSourceRuns)
```

4. 我们声明一个`_test_uuid`变量，其中包含要编译和运行的C代码段:

```cmake
set(_test_uuid
"
#include <uuid/uuid.h>
int main(int argc, char * argv[]) {
  uuid_t uuid;
  uuid_generate(uuid);
  return 0;
}
")
```

5. 我们声明`CMAKE_REQUIRED_LIBRARIES`变量后，对`check_c_source_runs`函数的调用。接下来，调用`check_c_source_runs`，其中测试代码作为第一个参数，`_runs`变量作为第二个参数，以保存执行的检查结果。之后，取消`CMAKE_REQUIRED_LIBRARIES`变量的设置:

```cmake
set(CMAKE_REQUIRED_LIBRARIES PkgConfig::UUID)
check_c_source_runs("${_test_uuid}" _runs)
unset(CMAKE_REQUIRED_LIBRARIES)
```

6. 如果检查没有成功，要么是代码段没有编译，要么是没有运行，我们会用致命的错误停止配置:

```cmake
if(NOT _runs)
	message(FATAL_ERROR "Cannot run a simple C executable using libuuid!")
endif()
```

7. 若成功，我们继续添加C++可执行文件作为目标，并链接到UUID:

```cmake
add_executable(use-uuid use-uuid.cpp)
target_link_libraries(use-uuid
  PUBLIC
  	PkgConfig::UUID
  )
```

### 工作原理

`check_<lang>_source_runs`用于C和C++的函数，与`check_<lang>_source_compile`相同，但在实际运行生成的可执行文件的地方需要添加一个步骤。对于`check_<lang>_source_compiles`, `check_<lang>_source_runs`的执行可以通过以下变量来进行:

- CMAKE_REQUIRED_FLAGS：设置编译器标志。
- CMAKE_REQUIRED_DEFINITIONS：设置预编译宏。
- CMAKE_REQUIRED_INCLUDES：设置包含目录列表。
- CMAKE_REQUIRED_LIBRARIES：设置可执行目标需要连接的库列表。

由于使用`pkg_search_module`生成的为导入目标，所以只需要将`CMAKE_REQUIRES_LIBRARIES`设置为`PkgConfig::UUID`，就可以正确设置包含目录。

正如`check_<lang>_source_compiles`是`try_compile`的包装器，`check_<lang>_source_runs`是CMake中另一个功能更强大的命令的包装器:`try_run`。因此，可以编写一个`CheckFortranSourceRuns.cmake`模块，通过适当包装`try_run`, 提供与C和C++模块相同的功能。

**NOTE**:*`pkg_search_module`只能定义导入目标(CMake 3.6),但目前的示例可以使工作，3.6之前版本的CMake可以通过手动设置所需的包括目录和库`check_c_source_runs`如下:`set(CMAKE_REQUIRED_INCLUDES $ {UUID_INCLUDE_DIRS})`和`set(CMAKE_REQUIRED_LIBRARIES $ {UUID_LIBRARIES})`。*

## 使用生成器表达式微调配置和编译

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-5/recipe-09 中找到，其中包含一个C++例子。该示例在CMake 3.9版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

CMake提供了一种特定于领域的语言，来描述如何配置和构建项目。自然会引入描述特定条件的变量，并在`CMakeLists.txt`中包含基于此的条件语句。

本示例中，我们将重新讨论生成器表达式。第4章中，以简洁地引用显式的测试可执行路径，使用了这些表达式。生成器表达式为逻辑和信息表达式，提供了一个强大而紧凑的模式，这些表达式在生成构建系统时进行评估，并生成特定于每个构建配置的信息。换句话说，生成器表达式用于引用仅在生成时已知，但在配置时未知或难于知晓的信息；对于文件名、文件位置和库文件后缀尤其如此。

本例中，我们将使用生成器表达式，有条件地设置预处理器定义，并有条件地链接到消息传递接口库(Message Passing Interface, MPI)，并允许我们串行或使用MPI构建相同的源代码。

**NOTE**:*本例中，我们将使用一个导入的目标来链接到MPI，该目标仅从CMake 3.9开始可用。但是，生成器表达式可以移植到CMake 3.0或更高版本。*

### 准备工作

我们将编译以下示例源代码(`example.cpp`):

```cmake
#include <iostream>

#ifdef HAVE_MPI
#include <mpi.h>
#endif
int main()
{
#ifdef HAVE_MPI
  // initialize MPI
  MPI_Init(NULL, NULL);

  // query and print the rank
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::cout << "hello from rank " << rank << std::endl;

  // initialize MPI
  MPI_Finalize();
#else
  std::cout << "hello from a sequential binary" << std::endl;
#endif /* HAVE_MPI */
}
```

代码包含预处理语句(`#ifdef HAVE_MPI ... #else ... #endif`)，这样我们就可以用相同的源代码编译一个顺序的或并行的可执行文件了。

### 具体实施

编写`CMakeLists.txt`文件时，我们将重用第3章第6节的一些构建块:

1. 声明一个C++11项目：

```cmake
cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(recipe-09 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

2. 然后，我们引入一个选项`USE_MPI`来选择MPI并行化，并将其设置为默认值`ON`。如果为`ON`，我们使用`find_package`来定位MPI环境:

```cmake
option(USE_MPI "Use MPI parallelization" ON)
if(USE_MPI)
	find_package(MPI REQUIRED)
endif()
```

3. 然后定义可执行目标，并有条件地设置相应的库依赖项(`MPI::MPI_CXX`)和预处理器定义(`HAVE_MPI`)，稍后将对此进行解释:

```cmake
add_executable(example example.cpp)
target_link_libraries(example
  PUBLIC
  	$<$<BOOL:${MPI_FOUND}>:MPI::MPI_CXX>
  )
target_compile_definitions(example
  PRIVATE
  	$<$<BOOL:${MPI_FOUND}>:HAVE_MPI>
  )
```

4. 如果找到MPI，还将打印由`FindMPI.cmake`导出的`INTERFACE_LINK_LIBRARIES`，为了方便演示，使用了`cmake_print_properties()`函数:

```cmake
if(MPI_FOUND)
  include(CMakePrintHelpers)
  cmake_print_properties(
    TARGETS MPI::MPI_CXX
    PROPERTIES INTERFACE_LINK_LIBRARIES
    )
endif()
```

5. 首先使用默认MPI配置。观察`cmake_print_properties()`的输出:

```bash
$ mkdir -p build_mpi
$ cd build_mpi
$ cmake ..

-- ...
--
Properties for TARGET MPI::MPI_CXX:
MPI::MPI_CXX.INTERFACE_LINK_LIBRARIES = "-Wl,-rpath -Wl,/usr/lib/openmpi -Wl,--enable-new-dtags -pthread;/usr/lib/openmpi/libmpi_cxx.so;/usr/lib/openmpi/libmpi.so"

```

6. 编译并运行并行例子:

```bash
$ cmake --build .
$ mpirun -np 2 ./example

hello from rank 0
hello from rank 1
```

7. 现在，创建一个新的构建目录，这次构建串行版本:

```bash
$ mkdir -p build_seq
$ cd build_seq
$ cmake -D USE_MPI=OFF ..
$ cmake --build .
$ ./example

hello from a sequential binary
```

### 工作原理

CMake分两个阶段生成项目的构建系统：配置阶段(解析`CMakeLists.txt`)和生成阶段(实际生成构建环境)。生成器表达式在第二阶段进行计算，可以使用仅在生成时才能知道的信息来调整构建系统。生成器表达式在交叉编译时特别有用，一些可用的信息只有解析`CMakeLists.txt`之后，或在多配置项目后获取，构建系统生成的所有项目可以有不同的配置，比如Debug和Release。

本例中，将使用生成器表达式有条件地设置链接依赖项并编译定义。为此，可以关注这两个表达式:

```cmake
target_link_libraries(example
  PUBLIC
  	$<$<BOOL:${MPI_FOUND}>:MPI::MPI_CXX>
  )
target_compile_definitions(example
  PRIVATE
  	$<$<BOOL:${MPI_FOUND}>:HAVE_MPI>
  )
```

如果`MPI_FOUND`为真，那么`$<BOOL:${MPI_FOUND}>`的值将为1。本例中，`$<$<BOOL:${MPI_FOUND}>:MPI::MPI_CXX>`将计算`MPI::MPI_CXX`，第二个生成器表达式将计算结果存在`HAVE_MPI`。如果将`USE_MPI`设置为`OFF`，则`MPI_FOUND`为假，两个生成器表达式的值都为空字符串，因此不会引入链接依赖关系，也不会设置预处理定义。

我们可以通过`if`来达到同样的效果:

```cmake
if(MPI_FOUND)
  target_link_libraries(example
    PUBLIC
    	MPI::MPI_CXX
    )
    
  target_compile_definitions(example
    PRIVATE
    	HAVE_MPI
    )
endif()
```

这个解决方案不太优雅，但可读性更好。我们可以使用生成器表达式来重新表达`if`语句，而这个选择取决于个人喜好。但当我们需要访问或操作文件路径时，生成器表达式尤其出色，因为使用变量和`if`构造这些路径可能比较困难。本例中，我们更注重生成器表达式的可读性。第4章中，我们使用生成器表达式来解析特定目标的文件路径。第11章中，我们会再次来讨论生成器。

### 更多信息

CMake提供了三种类型的生成器表达式:

- **逻辑表达式**，基本模式为`$<condition:outcome>`。基本条件为0表示false, 1表示true，但是只要使用了正确的关键字，任何布尔值都可以作为条件变量。
- **信息表达式**，基本模式为`$<information>`或`$<information:input>`。这些表达式对一些构建系统信息求值，例如：包含目录、目标属性等等。这些表达式的输入参数可能是目标的名称，比如表达式`$<TARGET_PROPERTY:tgt,prop>`，将获得的信息是tgt目标上的prop属性。
- **输出表达式**，基本模式为`$<operation>`或`$<operation:input>`。这些表达式可能基于一些输入参数，生成一个输出。它们的输出可以直接在CMake命令中使用，也可以与其他生成器表达式组合使用。例如, `- I$<JOIN:$<TARGET_PROPERTY:INCLUDE_DIRECTORIES>, -I>`将生成一个字符串，其中包含正在处理的目标的包含目录，每个目录的前缀由`-I`表示。

有关生成器表达式的完整列表，请参考https://cmake.org/cmake/help/latest/manual/cmake-generator-expressions.7.html





# CMake 完整使用教程 之七 生成源码

[CMake 完整使用教程 之七 生成源码 | 来唧唧歪歪(Ljjyy.com) - 多读书多实践，勤思考善领悟](https://www.ljjyy.com/archives/2021/03/100657)



本章的主要内容如下：

- 配置时生成源码
- 使用Python在配置时生成源码
- 构建时使用Python生成源码
- 记录项目版本信息以便报告
- 从文件中记录项目版本
- 配置时记录Git Hash值
- 构建时记录Git Hash值

大多数项目，使用版本控制跟踪源码。源代码通常作为构建系统的输入，将其转换为o文件、库或可执行程序。某些情况下，我们使用构建系统在配置或构建步骤时生成源代码。根据配置步骤中收集的信息，对源代码进行微调。另一个常用的方式，是记录有关配置或编译的信息，以保证代码行为可重现性。本章中，我们将演示使用CMake提供的源代码生成工具，以及各种相关的策略。

## 配置时生成源码

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-6/recipe-01 中找到，其中包含一个Fortran/C例子。该示例在CMake 3.10版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows(使用MSYS Makefiles)上进行过测试。*

代码生成在配置时发生，例如：CMake可以检测操作系统和可用库；基于这些信息，我们可以定制构建的源代码。本节和下面的章节中，我们将演示如何生成一个简单源文件，该文件定义了一个函数，用于报告构建系统配置。

### 准备工作

此示例的代码使用Fortran和C语言编写，第9章将讨论混合语言编程。主程序是一个简单的Fortran可执行程序，它调用一个C函数`print_info()`，该函数将打印配置信息。值得注意的是，在使用Fortran 2003时，编译器将处理命名问题(对于C函数的接口声明)，如示例所示。我们将使用的`example.f90`作为源文件:

```fortran
program hello_world

  implicit none
  
  interface
  	subroutine print_info() bind(c, name="print_info")
  	end subroutine
  end interface
  
  call print_info()
  
end program
```

C函数`print_info()`在模板文件`print_info.c.in`中定义。在配置时，以`@`开头和结尾的变量将被替换为实际值:

```c++
#include <stdio.h>
#include <unistd.h>

void print_info(void)
{
  printf("\n");
  printf("Configuration and build information\n");
  printf("-----------------------------------\n");
  printf("\n");
  printf("Who compiled | %s\n", "@_user_name@");
  printf("Compilation hostname | %s\n", "@_host_name@");
  printf("Fully qualified domain name | %s\n", "@_fqdn@");
  printf("Operating system | %s\n",
         "@_os_name@, @_os_release@, @_os_version@");
  printf("Platform | %s\n", "@_os_platform@");
  printf("Processor info | %s\n",
         "@_processor_name@, @_processor_description@");
  printf("CMake version | %s\n", "@CMAKE_VERSION@");
  printf("CMake generator | %s\n", "@CMAKE_GENERATOR@");
  printf("Configuration time | %s\n", "@_configuration_time@");
  printf("Fortran compiler | %s\n", "@CMAKE_Fortran_COMPILER@");
  printf("C compiler | %s\n", "@CMAKE_C_COMPILER@");
  printf("\n");

  fflush(stdout);
}
```

### 具体实施

在CMakeLists.txt中，我们首先必须对选项进行配置，并用它们的值替换`print_info.c.in`中相应的占位符。然后，将Fortran和C源代码编译成一个可执行文件:

1. 声明了一个Fortran-C混合项目:

```cmake
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(recipe-01 LANGUAGES Fortran C)
```

2. 使用`execute_process`为项目获取当且使用者的信息:

```cmake
execute_process(
  COMMAND
  	whoami
  TIMEOUT
  	1
  OUTPUT_VARIABLE
  	_user_name
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )
```

3. 使用`cmake_host_system_information()`函数(已经在第2章第5节遇到过)，可以查询很多系统信息:

```cmake
# host name information
cmake_host_system_information(RESULT _host_name QUERY HOSTNAME)
cmake_host_system_information(RESULT _fqdn QUERY FQDN)

# processor information
cmake_host_system_information(RESULT _processor_name QUERY PROCESSOR_NAME)
cmake_host_system_information(RESULT _processor_description QUERY PROCESSOR_DESCRIPTION)

# os information
cmake_host_system_information(RESULT _os_name QUERY OS_NAME)
cmake_host_system_information(RESULT _os_release QUERY OS_RELEASE)
cmake_host_system_information(RESULT _os_version QUERY OS_VERSION)
cmake_host_system_information(RESULT _os_platform QUERY OS_PLATFORM)
```

4. 捕获配置时的时间戳，并通过使用字符串操作函数:

```cmake
string(TIMESTAMP _configuration_time "%Y-%m-%d %H:%M:%S [UTC]" UTC)
```

5. 现在，准备好配置模板文件`print_info.c.in`。通过CMake的`configure_file`函数生成代码。注意，这里只要求以`@`开头和结尾的字符串被替换:

```cmake
configure_file(print_info.c.in print_info.c @ONLY)
```

6. 最后，我们添加一个可执行目标，并定义目标源：

```cmake
add_executable(example "")
target_sources(example
  PRIVATE
    example.f90
    ${CMAKE_CURRENT_BINARY_DIR}/print_info.c
  )
```

7. 下面是一个输出示例：

```bash
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
$ ./example

Configuration and build information
-----------------------------------
Who compiled | somebody
Compilation hostname | laptop
Fully qualified domain name | laptop
Operating system | Linux, 4.16.13-1-ARCH, #1 SMP PREEMPT Thu May 31 23:29:29 UTC 2018
Platform | x86_64
Processor info | Unknown P6 family, 2 core Intel(R) Core(TM) i5-5200U CPU @ 2.20GHz
CMake version | 3.11.3
CMake generator | Unix Makefiles
Configuration time | 2018-06-25 15:38:03 [UTC]
Fortran compiler | /usr/bin/f95
C compiler | /usr/bin/cc
```

### 工作原理

`configure_file`命令可以复制文件，并用变量值替换它们的内容。示例中，使用`configure_file`修改模板文件的内容，并将其复制到一个位置，然后将其编译到可执行文件中。如何调用`configure_file`:

```cmake
configure_file(print_info.c.in print_info.c @ONLY)
```

第一个参数是模板的名称为`print_info.c.in`。CMake假设输入文件的目录，与项目的根目录相对；也就是说，在`${CMAKE_CURRENT_SOURCE_DIR}/print_info.c.in`。我们选择`print_info.c`，作为第二个参数是配置文件的名称。假设输出文件位于相对于项目构建目录的位置：`${CMAKE_CURRENT_BINARY_DIR}/print_info.c`。

输入和输出文件作为参数时，CMake不仅将配置`@VAR@`变量，还将配置`${VAR}`变量。如果`${VAR}`是语法的一部分，并且不应该修改(例如在shell脚本中)，那么就很不方便。为了在引导CMake，应该将选项`@ONLY`传递给`configure_file`的调用，如前所述。

### 更多信息

注意，用值替换占位符时，CMake中的变量名应该与将要配置的文件中使用的变量名完全相同，并放在`@`之间。可以在调用`configure_file`时定义的任何CMake变量。我们的示例中，这包括所有内置的CMake变量，如`CMAKE_VERSION`或`CMAKE_GENERATOR`。此外，每当修改模板文件时，重新生成代码将触发生成系统的重新生成。这样，配置的文件将始终保持最新。

**TIPS**:*通过使用`CMake --help-variable-list`，可以从CMake手册中获得完整的内部CMake变量列表。*

**NOTE**:*`file(GENERATE…)`为提供了一个有趣的替代`configure_file`，这是因为`file`允许将生成器表达式作为配置文件的一部分进行计算。但是，每次运行CMake时，`file(GENERATE…)`都会更新输出文件，这将强制重新构建依赖于该输出的所有目标。详细可参见https://crascit.com/2017/04/18/generated-sources-in-cmake-build 。*

## 使用Python在配置时生成源码

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-6/recipe-02 中找到，其中包含一个Fortran/C例子。该示例在CMake 3.10版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows(使用MSYS Makefile)上进行过测试。*

本示例中，我们将再次从模板`print_info.c.in`生成`print_info.c`。但这一次，将假设CMake函数`configure_file()`没有创建源文件，然后使用Python脚本模拟这个过程。当然，对于实际的项目，我们可能更倾向于使用`configure_file()`，但有时使用Python生成源代码的需要时，我们也应该知道如何应对。

这个示例有严重的限制，不能完全模拟`configure_file()`。我们在这里介绍的方法，不能生成一个自动依赖项，该依赖项将在构建时重新生成`print_info.c`。换句话说，如果在配置之后删除生成的`print_info.c`，则不会重新生成该文件，构建也会失败。要正确地模拟`configure_file()`，需要使用`add_custom_command()`和`add_custom_target()`。我们将在第3节中使用它们，来克服这个限制。

这个示例中，我们将使用一个简单的Python脚本。这个脚本将读取`print_info.c.in`。用从CMake传递给Python脚本的参数替换文件中的占位符。对于更复杂的模板，我们建议使用外部工具，比如Jinja(参见[http://jinja.pocoo.org](http://jinja.pocoo.org/) )。

```python
def configure_file(input_file, output_file, vars_dict):

  with input_file.open('r') as f:
  	template = f.read()

  for var in vars_dict: 
  	template = template.replace('@' + var + '@', vars_dict[var])

  with output_file.open('w') as f:
  	f.write(template)
```

这个函数读取一个输入文件，遍历`vars_dict`变量中的目录，并用对应的值替换`@key@`，再将结果写入输出文件。这里的键值对，将由CMake提供。

### 准备工作

`print_info.c.in`和`example.f90`与之前的示例相同。此外，我们将使用Python脚本`configurator.py`，它提供了一个函数:

```cmake
def configure_file(input_file, output_file, vars_dict):
  with input_file.open('r') as f:
  	template = f.read()
    
  for var in vars_dict:
  	template = template.replace('@' + var + '@', vars_dict[var])
    
  with output_file.open('w') as f:
  	f.write(template)
```

该函数读取输入文件，遍历`vars_dict`字典的所有键，用对应的值替换模式`@key@`，并将结果写入输出文件(键值由CMake提供)。

### 具体实施

与前面的示例类似，我们需要配置一个模板文件，但这一次，使用Python脚本模拟`configure_file()`函数。我们保持CMakeLists.txt基本不变，并提供一组命令进行替换操作`configure_file(print_info.c.in print_info.c @ONLY)`，接下来将逐步介绍这些命令:

1. 首先，构造一个变量`_config_script`，它将包含一个Python脚本，稍后我们将执行这个脚本:

```cmake
set(_config_script
"
from pathlib import Path
source_dir = Path('${CMAKE_CURRENT_SOURCE_DIR}')
binary_dir = Path('${CMAKE_CURRENT_BINARY_DIR}')
input_file = source_dir / 'print_info.c.in'
output_file = binary_dir / 'print_info.c'

import sys
sys.path.insert(0, str(source_dir))

from configurator import configure_file
vars_dict = {
  '_user_name': '${_user_name}',
  '_host_name': '${_host_name}',
  '_fqdn': '${_fqdn}',
  '_processor_name': '${_processor_name}',
  '_processor_description': '${_processor_description}',
  '_os_name': '${_os_name}',
  '_os_release': '${_os_release}',
  '_os_version': '${_os_version}',
  '_os_platform': '${_os_platform}',
  '_configuration_time': '${_configuration_time}',
  'CMAKE_VERSION': '${CMAKE_VERSION}',
  'CMAKE_GENERATOR': '${CMAKE_GENERATOR}',
  'CMAKE_Fortran_COMPILER': '${CMAKE_Fortran_COMPILER}',
  'CMAKE_C_COMPILER': '${CMAKE_C_COMPILER}',
}
configure_file(input_file, output_file, vars_dict)
")
```

2. 使用`find_package`让CMake使用Python解释器:

```cmake
find_package(PythonInterp QUITE REQUIRED)
```

3. 如果找到Python解释器，则可以在CMake中执行`_config_script`，并生成`print_info.c`文件:

```cmake
execute_process(
  COMMAND
  	${PYTHON_EXECUTABLE} "-c" ${_config_script}
  )
```

4. 之后，定义可执行目标和依赖项，这与前一个示例相同。所以，得到的输出没有变化。

### 工作原理

回顾一下对CMakeLists.txt的更改。

我们执行了一个Python脚本生成`print_info.c`。运行Python脚本前，首先检测Python解释器，并构造Python脚本。Python脚本导入`configure_file`函数，我们在`configurator.py`中定义了这个函数。为它提供用于读写的文件位置，并将其值作为键值对。

此示例展示了生成配置的另一种方法，将生成任务委托给外部脚本，可以将配置报告编译成可执行文件，甚至库目标。我们在前面的配置中认为的第一种方法更简洁，但是使用本示例中提供的方法，我们可以灵活地使用Python(或其他语言)，实现任何在配置时间所需的步骤。使用当前方法，我们可以通过脚本的方式执行类似`cmake_host_system_information()`的操作。

但要记住，这种方法也有其局限性，它不能在构建时重新生成`print_info.c`的自动依赖项。下一个示例中，我们应对这个挑战。

### 更多信息

我们可以使用`get_cmake_property(_vars VARIABLES)`来获得所有变量的列表，而不是显式地构造`vars_dict`(这感觉有点重复)，并且可以遍历`_vars`的所有元素来访问它们的值:

```cmake
get_cmake_property(_vars VARIABLES)
foreach(_var IN ITEMS ${_vars})
  message("variable ${_var} has the value ${${_var}}") 
endforeach()
```

使用这种方法，可以隐式地构建`vars_dict`。但是，必须注意转义包含字符的值，例如:`;`， Python会将其解析为一条指令的末尾。

## 构建时使用Python生成源码

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-6/recipe-03 中找到，其中包含一个C++例子。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

构建时根据某些规则生成冗长和重复的代码，同时避免在源代码存储库中显式地跟踪生成的代码生成源代码，是开发人员工具箱中的一个重要工具，例如：根据检测到的平台或体系结构生成不同的源代码。或者，可以使用Python，根据配置时收集的输入，在构建时生成高效的C++代码。其他生成器解析器，比如：Flex (https://github.com/westes/flex )和Bison(https://www.gnu.org/software/bison/ )；元对象编译器，如Qt的moc(http://doc.qt.io/qt5/moc.html )；序列化框架，如谷歌的protobuf (https://developers.google.com/protocol-buffers/ )。

### 准备工作

为了提供一个具体的例子，我们需要编写代码来验证一个数字是否是质数。现在有很多算法，例如：可以用埃拉托色尼的筛子(sieve of Eratosthenes)来分离质数和非质数。如果有很多验证数字，我们不希望对每一个数字都进行Eratosthenes筛选。我们想要做的是将所有质数一次制表，直到数字的上限，然后使用一个表查的方式，找来验证大量的数字。

本例中，将在编译时使用Python为查找表(质数向量)生成C++代码。当然，为了解决这个特殊的编程问题，我们还可以使用C++生成查询表，并且可以在运行时执行查询。

让我们从`generate.py`脚本开始。这个脚本接受两个命令行参数——一个整数范围和一个输出文件名:

```python
"""
Generates C++ vector of prime numbers up to max_number
using sieve of Eratosthenes.
"""
import pathlib
import sys

# for simplicity we do not verify argument list
max_number = int(sys.argv[-2])
output_file_name = pathlib.Path(sys.argv[-1])

numbers = range(2, max_number + 1)
is_prime = {number: True for number in numbers}

for number in numbers:
  current_position = number
  if is_prime[current_position]:
    while current_position <= max_number:
      current_position += number
      is_prime[current_position] = False
      
primes = (number for number in numbers if is_prime[number])

code = """#pragma once

#include <vector>

const std::size_t max_number = {max_number};
std::vector<int> & primes() {{
  static std::vector<int> primes;
  {push_back}
  return primes;
}}
"""
push_back = '\n'.join([' primes.push_back({:d});'.format(x) for x in primes])
output_file_name.write_text(
code.format(max_number=max_number, push_back=push_back))
```

我们的目标是生成一个`primes.hpp`，并将其包含在下面的示例代码中:

```cpp
#include "primes.hpp"

#include <iostream>
#include <vector>

int main() {
  std::cout << "all prime numbers up to " << max_number << ":";
  
  for (auto prime : primes())
  	std::cout << " " << prime;
  
  std::cout << std::endl;
  
  return 0;
}
```

### 具体实施

下面是CMakeLists.txt命令的详解:

1. 首先，定义项目并检测Python解释器:

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(recipe-03 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(PythonInterp QUIET REQUIRED)
```

2. 将生成的代码放在`${CMAKE_CURRENT_BINARY_DIR}/generate`下，需要告诉CMake创建这个目录:

```cmake
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated)
```

3. Python脚本要求质数的上限，使用下面的命令，我们可以设置一个默认值:

```cmake
set(MAX_NUMBER "100" CACHE STRING "Upper bound for primes")
```

4. 接下来，定义一个自定义命令来生成头文件:

```cmake
add_custom_command(
  OUTPUT
  	${CMAKE_CURRENT_BINARY_DIR}/generated/primes.hpp
  COMMAND
  	${PYTHON_EXECUTABLE} generate.py ${MAX_NUMBER} 	${CMAKE_CURRENT_BINARY_DIR}/generated/primes.hpp
  WORKING_DIRECTORY
  	${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS
  	generate.py
)
```

5. 最后，定义可执行文件及其目标，包括目录和依赖关系:

```cmake
add_executable(example "")
target_sources(example
  PRIVATE
  	example.cpp
  	${CMAKE_CURRENT_BINARY_DIR}/generated/primes.hpp
  )
target_include_directories(example
  PRIVATE
  	${CMAKE_CURRENT_BINARY_DIR}/generated
  )
```

6. 准备测试:

```bash
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
$ ./example
all prime numbers up to 100: 2 3 5 7 11 13 17 19 23 29 31 37 41 43 47 53 59 61 67 71 73 79
```

### 具体实施

为了生成头文件，我们定义了一个自定义命令，它执行`generate.py`脚本，并接受`${MAX_NUMBER}`和文件路径(`${CMAKE_CURRENT_BINARY_DIR}/generated/primes.hpp`)作为参数:

```cmake
add_custom_command(
  OUTPUT
  	${CMAKE_CURRENT_BINARY_DIR}/generated/primes.hpp
  COMMAND
  	${PYTHON_EXECUTABLE} generate.py ${MAX_NUMBER} ${CMAKE_CURRENT_BINARY_DIR}/generated/primes.hpp
  WORKING_DIRECTORY
  	${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS
  	generate.py
  )
```

为了生成源代码，我们需要在可执行文件的定义中，使用`target_sources`很容易实现添加源代码作为依赖项:

```cmake
target_sources(example
  PRIVATE
  	example.cpp
  	${CMAKE_CURRENT_BINARY_DIR}/generated/primes.hpp
  )
```

前面的代码中，我们不需要定义新的目标。头文件将作为示例的依赖项生成，并在每次`generate.py`脚本更改时重新生成。如果代码生成脚本生成多个源文件，那么要将所有生成的文件列出，做为某些目标的依赖项。

### 更多信息

我们提到所有的生成文件，都应该作为某个目标的依赖项。但是，我们可能不知道这个文件列表，因为它是由生成文件的脚本决定的，这取决于我们提供给配置的输入。这种情况下，我们可能会尝试使用`file(GLOB…)`将生成的文件收集到一个列表中(参见https://cmake.org/cmake/help/v3.5/command/file.html )。

`file(GLOB…)`在配置时执行，而代码生成是在构建时发生的。因此可能需要一个间接操作，将`file(GLOB…)`命令放在一个单独的CMake脚本中，使用`${CMAKE_COMMAND} -P`执行该脚本，以便在构建时获得生成的文件列表。

## 记录项目版本信息以便报告

**NOTE**:*此示例代码可以在 https://github.com/dev-cafe/cmake-cookbook/tree/v1.0/chapter-6/recipe-04 中找到，其中包含一个C和Fortran例子。该示例在CMake 3.5版(或更高版本)中是有效的，并且已经在GNU/Linux、macOS和Windows上进行过测试。*

代码版本很重要，不仅是为了可重复性，还为了记录API功能或简化支持请求和bug报告。源代码通常处于某种版本控制之下，例如：可以使用Git标记附加额外版本号(参见[https://semver.org](https://semver.org/) )。然而，不仅需要对源代码进行版本控制，而且可执行文件还需要记录项目版本，以便将其打印到代码输出或用户界面上。

本例中，将在CMake源文件中定义版本号。我们的目标是在配置项目时将程序版本记录到头文件中。然后，生成的头文件可以包含在代码的正确位置和时间，以便将代码版本打印到输出文件或屏幕上。

### 准备工作

将使用以下C文件(`example.c`)打印版本信息:

```cpp
#include "version.h"

#include <stdio.h>

int main() {
  printf("This is output from code %s\n", PROJECT_VERSION);
  printf("Major version number: %i\n", PROJECT_VERSION_MAJOR);
  printf("Minor version number: %i\n", PROJECT_VERSION_MINOR);
  
  printf("Hello CMake world!\n");
}
```

这里，假设`PROJECT_VERSION_MAJOR`、`PROJECT_VERSION_MINOR`和`PROJECT_VERSION`是在`version.h`中定义的。目标是从以下模板中生成`version.h.in`:

```ini
#pragma once

#define PROJECT_VERSION_MAJOR @PROJECT_VERSION_MAJOR@
#define PROJECT_VERSION_MINOR @PROJECT_VERSION_MINOR@
#define PROJECT_VERSION_PATCH @PROJECT_VERSION_PATCH@

#define PROJECT_VERSION "v@PROJECT_VERSION@"
```

这里使用预处理器定义，也可以使用字符串或整数常量来提高类型安全性(稍后我们将对此进行演示)。从CMake的角度来看，这两种方法是相同的。

### 如何实施

我们将按照以下步骤，在模板头文件中对版本进行注册:

1. 要跟踪代码版本，我们可以在CMakeLists.txt中调用CMake的`project`时定义项目版本:

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(recipe-04 VERSION 2.0.1 LANGUAGES C)
```







































































































































