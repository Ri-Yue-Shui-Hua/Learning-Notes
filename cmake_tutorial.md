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



## 在编译时复制文件方法

[CMake 复制文件方法 - JoyPoint - 博客园 (cnblogs.com)](https://www.cnblogs.com/JoyPoint/p/11629521.html)

我们经常会遇到将第三方库文件复制到项目运行时文件夹，或者将子项目生成的库文件复制到项目运行时文件夹的情况，本文介绍FILE-COPY、add_custom_command、ADD_CUSTOM_TARGET三种方法及CMake COMMAND提供的命令说明。

### FILE-COPY



```cmake
file(<COPY|INSTALL> <files>... DESTINATION <dir>
     [FILE_PERMISSIONS <permissions>...]
     [DIRECTORY_PERMISSIONS <permissions>...]
     [NO_SOURCE_PERMISSIONS] [USE_SOURCE_PERMISSIONS]
     [FOLLOW_SYMLINK_CHAIN]
     [FILES_MATCHING]
     [[PATTERN <pattern> | REGEX <regex>]
     [EXCLUDE] [PERMISSIONS <permissions>...]] [...])
```



COPY将文件，目录和符号链接复制到目标文件夹。相对于当前源目录评估相对输入路径，相对于当前构建目录评估相对目的地。复制会保留输入文件的时间戳，并优化文件（如果该文件存在于具有相同时间戳的目标文件中）。复制将保留输入权限，除非给出明确的权限或NO_SOURCE_PERMISSIONS（默认为USE_SOURCE_PERMISSIONS）。

如果指定了FOLLOW_SYMLINK_CHAIN，则COPY将在给定的路径上递归解析符号链接，直到找到真实文件为止，然后在目标位置为遇到的每个符号链接安装相应的符号链接。对于已安装的每个符号链接，解析都会从目录中剥离，仅保留文件名，这意味着新符号链接指向与符号链接相同目录中的文件。此功能在某些Unix系统上很有用，在这些系统中，库是作为带有符号链接的版本号安装的，而较少特定的版本指向的是特定版本。FOLLOW_SYMLINK_CHAIN会将所有这些符号链接和库本身安装到目标目录中。例如，如果您具有以下目录结构：

- /opt/foo/lib/libfoo.so.1.2.3
- /opt/foo/lib/libfoo.so.1.2 -> libfoo.so.1.2.3
- /opt/foo/lib/libfoo.so.1 -> libfoo.so.1.2
- /opt/foo/lib/libfoo.so -> libfoo.so.1

你可以：

```cmake
file(COPY /opt/foo/lib/libfoo.so DESTINATION lib FOLLOW_SYMLINK_CHAIN)
```

这会将所有符号链接和libfoo.so.1.2.3本身安装到lib中。

请参阅install（DIRECTORY）命令以获取权限，FILES_MATCHING，PATTERN，REGEX和EXCLUDE选项的文档。即使使用选项来选择文件的子集，复制目录也会保留其内容的结构。

INSTALL与COPY略有不同：它打印状态消息（取决于CMAKE_INSTALL_MESSAGE变量），并且默认为NO_SOURCE_PERMISSIONS。 install（）命令生成的安装脚本使用此签名（以及一些未记录的内部使用选项）。



### ADD_CUSTOM_COMMAND

add_custom_command：

该命令可以为生成的构建系统添加一条自定义的构建规则。这里又包含两种使用方式，一种是通过自定义命令在构建中生成输出文件，另外一种是向构建目标添加自定义命令。命令格式分别为：

(1)生成文件

```cmake
add_custom_command(OUTPUT output1 [output2 ...]
                   COMMAND command1 [ARGS] [args1...]
                   [COMMAND command2 [ARGS] [args2...] ...]
                   [MAIN_DEPENDENCY depend]
                   [DEPENDS [depends...]]
                   [BYPRODUCTS [files...]]
                   [IMPLICIT_DEPENDS <lang1> depend1
                                    [<lang2> depend2] ...]
                   [WORKING_DIRECTORY dir]
                   [COMMENT comment]
                   [DEPFILE depfile]
                   [JOB_POOL job_pool]
                   [VERBATIM] [APPEND] [USES_TERMINAL]
                   [COMMAND_EXPAND_LISTS])
```



参数介绍：

- OUTPUT：

指定命令预期产生的输出文件。如果输出文件的名称是相对路径，即相对于当前的构建的源目录路径。输出文件可以指定多个output1,output2(可选)等。

-  COMMAND：

指定要在构建时执行的命令行。如果指定多个COMMAND，它们将按顺序执行。ARGS参数是为了向后兼容，为可选参数。args1和args2为参数，多个参数用空格隔开。

-  MAIN_DEPENDENCY：

可选命令，指定命令的主要输入源文件。

- DEPENDS：

指定命令所依赖的文件。

- BYPRODUCTS：

可选命令，指定命令预期产生的文件，但其修改时间可能会比依赖性更新，也可能不会更新。

- IMPLICIT_DEPENDS：

可选命令，请求扫描输入文件的隐式依赖关系。给定的语言指定应使用相应的依赖性扫描器的编程语言。目前只支持C和CXX语言扫描器。必须为IMPLICIT_DEPENDS列表中的每个文件指定语言。从扫描中发现的依赖关系在构建时添加到自定义命令的依赖关系。请注意，IMPLICIT_DEPENDS选项目前仅支持Makefile生成器，并且将被其他生成器忽略。

- WORKING_DIRECTORY：

可选命令，使用给定的当前工作目录执行命令。如果它是相对路径，它将相对于对应于当前源目录的构建树目录。

- COMMENT：

可选命令，在构建时执行命令之前显示给定消息。

- DEPFILE：

可选命令，为Ninja生成器指定一个.d depfile。 .d文件保存通常由自定义命令本身发出的依赖关系。对其他生成器使用DEPFILE是一个错误。

- COMMAND_EXPAND_LISTS：

将扩展COMMAND参数中的列表，包括使用生成器表达式创建的列表，从而允许COMMAND参数，例如${CC} "-I$<JOIN:$<TARGET_PROPERTY:foo,INCLUDE_DIRECTORIES>,;-I>" foo.cc为适当扩展。

 

VERBATIM：

对于构建工具，将正确转义命令的所有自变量，以便调用的命令不变地接收每个自变量。 请注意，在add_custom_command甚至没有看到参数之前，CMake语言处理器仍使用一种转义。 建议使用VERBATIM，因为它可以使行为正确。 如果未提供VERBATIM，则该行为是特定于平台的，因为没有针对工具的特殊字符的保护。

 使用实例：

```cmake
add_executable(MakeTable MakeTable.cxx)
add_custom_command (
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/Table.h
  COMMAND MakeTable ${CMAKE_CURRENT_BINARY_DIR}/Table.h
  DEPENDS MakeTable
  COMMENT "This is a test"
  )
```



(2)自定义构建事件

```cmake
add_custom_command(TARGET <target>
                   PRE_BUILD | PRE_LINK | POST_BUILD
                   COMMAND command1 [ARGS] [args1...]
                   [COMMAND command2 [ARGS] [args2...] ...]
                   [BYPRODUCTS [files...]]
                   [WORKING_DIRECTORY dir]
                   [COMMENT comment]
                   [VERBATIM] [USES_TERMINAL])
```



参数介绍：

- TARGET：

定义了与构建指定相关联的新命令。当已经存在是，相应的command将不再执行。

- PRE_BUILD：

在目标中执行任何其他规则之前运行。这仅在Visual Studio 7或更高版本上受支持。对于所有其他生成器PRE_BUILD将被视为PRE_LINK。

- PRE_LINK：

在编译源之后运行，但在链接二进制文件或运行静态库的库管理器或存档器工具之前运行。

- POST_BUILD：

在目标中的所有其他规则都已执行后运行。

**示例：将子项目生成的库文件复制到项目运行时文件夹。**

```cmake
#=============Copy Plugins Runtime FILES to Main Project============ 
#一般将此内容放在lib子项目的CMakelists.txt的最后，该方法采用POST_BUILD，所以需注意要复制的源应该是一个固定字符串，而不能用FILE GLOB的方法，因为在编译前该源为空，在VS中会出现MSB3073错误，提示copy from 为空值。
SET(Plugins_TEST_Debug_DLL_FILE  
    ${CMAKE_CURRENT_BINARY_DIR}/Debug/lib${PROJECT_NAME}.dll 
) 
SET(Plugins_TEST_Release_DLL_FILE  
    ${CMAKE_CURRENT_BINARY_DIR}/Release/lib${PROJECT_NAME}.dll 
) 
   
add_custom_command(TARGET ${PROJECT_NAME} 
   POST_BUILD 
   COMMAND ${CMAKE_COMMAND} -E 
       copy_if_different  
        "$<$<CONFIG:Release>:${Plugins_TEST_Release_DLL_FILE}>"  
        "$<$<CONFIG:Debug>:${Plugins_TEST_Debug_DLL_FILE}>"  
        "${CMAKE_BINARY_DIR}/$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>/Plugins/org_test_plugins/" 
) 
```



### ADD_CUSTOM_TARGET

add_custom_target：

该命令可以给指定名称的目标执行指定的命令，该目标没有输出文件，并始终被构建。命令的格式为：

```cmake
add_custom_target(Name [ALL] [command1 [args1...]]
                  [COMMAND command2 [args2...] ...]
                  [DEPENDS depend depend depend ... ]
                  [BYPRODUCTS [files...]]
                  [WORKING_DIRECTORY dir]
                  [COMMENT comment]
                  [JOB_POOL job_pool]
                  [VERBATIM] [USES_TERMINAL]
                  [COMMAND_EXPAND_LISTS]
                  [SOURCES src1 [src2...]])
```

参数介绍：

- Name：

指定目标的名称，单独成为一个子项目。

- ALL：

表明此目标应添加到默认构建目标，以便每次都将运行（该命令名称不能为ALL）

- SOURCES：

指定要包括在自定义目标中的其他源文件。指定的源文件将被添加到IDE项目文件中，以方便编辑，即使它们没有构建规则。



**示例：将第三方库文件复制到项目运行时文件夹**

```cmake
#=============Copy Source files to Build Runtime Dir=============== 
#该内容一般放在项目顶层CMakelists.txt的最后，
#目的是将项目生成后的执行文件所需的第三方库复制到执行程序目录，
#并区分Debug和Release版本。
#该方法中的COMMAND_EXPAND_LISTS参数值得关注，可以复制列表内所有文件。
FILE(GLOB Plugin_Runtime_Debug_DLL_FILES CONFIGURE_DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/Plugin_Runtime_Dir/Debug/*.* 
) 
FILE(GLOB Plugin_Runtime_Release_DLL_FILES CONFIGURE_DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/Plugin_Runtime_Dir/Release/*.* 
) 
FILE(GLOB Plugin_Runtime_Debug_Resources_FILES CONFIGURE_DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/Plugin_Runtime_Dir/Debug/Resources/icos/*.* 
) 
FILE(GLOB Plugin_Runtime_Release_Resources_FILES CONFIGURE_DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/Plugin_Runtime_Dir/Release/Resources/icos/*.* 
) 
   
add_custom_target(CopyRuntimeFiles ALL 
  VERBATIM 
  COMMAND_EXPAND_LISTS 
  COMMAND ${CMAKE_COMMAND} -E 
          make_directory "${PROJECT_BINARY_DIR}/$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>/" 
COMMAND ${CMAKE_COMMAND} -E 
         copy_if_different  
                "$<$<CONFIG:Release>:${Plugin_Runtime_Release_DLL_FILES}>"  
                 "$<$<CONFIG:Debug>:${Plugin_Runtime_Debug_DLL_FILES}>" 
                "${PROJECT_BINARY_DIR}/$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>/"   
COMMAND ${CMAKE_COMMAND} -E 
        make_directory "${PROJECT_BINARY_DIR}/$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>/Resources/icos/" 
COMMAND ${CMAKE_COMMAND} -E 
        copy_if_different  
               "$<$<CONFIG:Release>:${Plugin_Runtime_Release_Resources_FILES}>"  
               "$<$<CONFIG:Debug>:${Plugin_Runtime_Debug_Resources_FILES}>" 
               "${PROJECT_BINARY_DIR}/$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>/Resources/icos/" 
COMMAND ${CMAKE_COMMAND} -E 
        make_directory "${PROJECT_BINARY_DIR}/$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>/Plugins/org_test_plugins/" 
) 
```













































