# C++ 模板

https://www.runoob.com/cplusplus/cpp-templates.html

模板是泛型编程的基础，泛型编程即以一种独立于任何特定类型的方式编写代码。

模板是创建泛型类或函数的蓝图或公式。库容器，比如迭代器和算法，都是泛型编程的例子，它们都使用了模板的概念。

每个容器都有一个单一的定义，比如 **向量**，我们可以定义许多不同类型的向量，比如 **vector <int>** 或 **vector <string>**。

您可以使用模板来定义函数和类，接下来让我们一起来看看如何使用。

## 函数模板

模板函数定义的一般形式如下所示：

```cpp
template <typename type> ret-type func-name(parameter list)
{
   // 函数的主体
}
```

在这里，type 是函数所使用的数据类型的占位符名称。这个名称可以在函数定义中使用。

下面是函数模板的实例，返回两个数中的最大值：

```cpp
#include <iostream>
#include <string>
 
using namespace std;
 
template <typename T>
inline T const& Max (T const& a, T const& b) 
{ 
    return a < b ? b:a; 
} 
int main ()
{
 
    int i = 39;
    int j = 20;
    cout << "Max(i, j): " << Max(i, j) << endl; 
 
    double f1 = 13.5; 
    double f2 = 20.7; 
    cout << "Max(f1, f2): " << Max(f1, f2) << endl; 
 
    string s1 = "Hello"; 
    string s2 = "World"; 
    cout << "Max(s1, s2): " << Max(s1, s2) << endl; 
 
    return 0;
}
```

当上面的代码被编译和执行时，它会产生下列结果：

```bash
Max(i, j): 39
Max(f1, f2): 20.7
Max(s1, s2): World
```

## 类模板

正如我们定义函数模板一样，我们也可以定义类模板。泛型类声明的一般形式如下所示：

```cpp
template <class type> class class-name {
.
.
.
}
```

在这里，**type** 是占位符类型名称，可以在类被实例化的时候进行指定。您可以使用一个逗号分隔的列表来定义多个泛型数据类型。

下面的实例定义了类 Stack<>，并实现了泛型方法来对元素进行入栈出栈操作：

```cpp
#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>
#include <stdexcept>
 
using namespace std;
 
template <class T>
class Stack { 
  private: 
    vector<T> elems;     // 元素 
 
  public: 
    void push(T const&);  // 入栈
    void pop();               // 出栈
    T top() const;            // 返回栈顶元素
    bool empty() const{       // 如果为空则返回真。
        return elems.empty(); 
    } 
}; 
 
template <class T>
void Stack<T>::push (T const& elem) 
{ 
    // 追加传入元素的副本
    elems.push_back(elem);    
} 
 
template <class T>
void Stack<T>::pop () 
{ 
    if (elems.empty()) { 
        throw out_of_range("Stack<>::pop(): empty stack"); 
    }
    // 删除最后一个元素
    elems.pop_back();         
} 
 
template <class T>
T Stack<T>::top () const 
{ 
    if (elems.empty()) { 
        throw out_of_range("Stack<>::top(): empty stack"); 
    }
    // 返回最后一个元素的副本 
    return elems.back();      
} 
 
int main() 
{ 
    try { 
        Stack<int>         intStack;  // int 类型的栈 
        Stack<string> stringStack;    // string 类型的栈 
 
        // 操作 int 类型的栈 
        intStack.push(7); 
        cout << intStack.top() <<endl; 
 
        // 操作 string 类型的栈 
        stringStack.push("hello"); 
        cout << stringStack.top() << std::endl; 
        stringStack.pop(); 
        stringStack.pop(); 
    } 
    catch (exception const& ex) { 
        cerr << "Exception: " << ex.what() <<endl; 
        return -1;
    } 
}
```

当上面的代码被编译和执行时，它会产生下列结果：

```bash
7
hello
Exception: Stack<>::pop(): empty stack

--------------------------------
Process exited after 0.07319 seconds with return value 4294967295

Press ANY key to exit...
```



## 获取时间

```cpp
#include<iostream>
#include<ctime>

using namespace std;

int main()
{
	//基于当前系统的当前时间
	time_t now=time(0);
	
	//把now转换为字符串类型
	char *dt=ctime(&now);
	
	cout<<"本地日期和时间: "<<dt<<endl;
	
	//把now转换为tm结构
	tm *gmtm=gmtime(&now);
	dt=asctime(gmtm);
	cout<<"UTC日期和时间: "<<dt<<endl;
	return 0;
}
```

运行输出：

```bash
本地日期和时间: Wed Aug 24 20:10:28 2022

UTC日期和时间: Wed Aug 24 12:10:28 2022


--------------------------------
Process exited after 0.007458 seconds with return value 0

Press ANY key to exit...
```



## 随机数生成

C++中有专门的随机数生成器

```cpp
#include<cstdlib>

int rand();
```

于是我们激动地实验了一下这个函数
发现了一个严肃的问题：为什么每次“随机”出来的数都是一样的呢？？？

这就引发我们思考另一个很重要地问题：

计算机可以产生真正的随机数嘛？
很遗憾，不行。。。
所谓的随机数，实际上是事先存储在计算机内部的一个数表
为了使取出来的数**可视为随机**，我们需要给计算机一个**种子**

根据种子为基准以某个递推公式推算出一系列数，因为其周期长，故在一定范围内可以看成是随机数

怎么设置种子
我们需要在使用随机数生成函数之前调用一次：

```cpp
void srand(unsigned int seed);
```

通过设置seed来产生不同的随机数，只要种子不同，那么通过rand()得到的随机数序列就是不同的；反之，如果种子是一样的，那么通过rand()得到的随机数序列就是相同的

那么我们**如何选择种子就显得无比重要**
一般的，我们就选择**time(0)**



```cpp
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

int main()
{
	const int seq_int = 41;//定义一个常量值为41
	int random_val = 0;//用来存放随机数
	srand((unsigned int)time(NULL));//srand中的参数就是seed，也就是随机数将会在0~(seed-1)中进行产生
	
	random_val = rand() % seq_int + 60;//产生0~100以内的数据
	
	cout << random_val << endl;
	
	//system("pause");
	return 0;
}

```

```bash
77

--------------------------------
Process exited after 0.006921 seconds with return value 0

Press ANY key to exit...
```



测试，生成随机数的时间确实不快。



## 常见关键字

### const

[C++ const 关键字小结 | 菜鸟教程 (runoob.com)](https://www.runoob.com/w3cnote/cpp-const-keyword.html)

[C语言中const的用法详解 (qq.com)](https://mp.weixin.qq.com/s/xFrmugLMMK3-QTklTgQMNg)

#### const基本介绍

const是constant的简写，用来定义常量，它限定一个变量不允许被改变，产生静态作用。const最开始推出的目的是为了取代预编译指令，取长补短。

#### 与define的对比

1. define是预编译指令，定义的宏是在预处理阶段展开的，而const是普通变量的定义，是只读变量，且是在编译运行阶段使用的。
2. define定义的是常量，define定义的宏在编译后消失了，它不占用内存，而const定义的常变量本质上仍然是一个变量，具有变量的基本属性，有类型、占用存储单元，除了不能作为数组的长度，用const定义的常变量具有宏的优点，而且使用更方便。
3. define定义的对象没有数据类型，编译器只能机械地进行字符替换，没有类型安全检查，即会出现“`边际问题`”或者是“`括号问题`”。而const定义的是变量，有数据类型。

#### 修饰局部变量

```
const int num=5;
int const num=5;
```

这两种写法是一样的，都是表示变量num的值不能被改变，用const修饰变量时，一定要初始化，否则之后就不能再进行赋值了（后面会讲到一种特殊情况）。

接下来看看const用于修饰常量静态字符串，例如：

```
const char* str="techdreamer";
```

如果没有const的修饰，我们可能会在后不经意间的修改代码，比如`str[4]=’D’`，这样会导致对只读内存区域的赋值，使程序异常终止。

而加上const修饰之后，这个错误就能在程序被编译的时候立即被检查出来，让逻辑错误在编译期被发现，避免我们在后续中继续debug。

#### 修饰全局变量

全局变量的作用域是整个文件，且全局变量的生存周期为程序运行的整个过程，所以我们应该尽量避免使用全局变量，一旦某个函数改变了全局变量的值，会影响到其他引用这个变量的函数，是一个很隐蔽的操作。

如果一定要用全局变量，应该尽量的使用const进行修饰，防止不必要的人为修改，使用 const 修饰过的局部变量就有了静态特性，它的生存周期也是程序运行的整个过程，虽然有了静态特性，但并不是说它变成了静态变量。

#### 修饰常量指针与指针常量

##### 常量指针



常量指针是指针指向的内容是常量，可以有以下两种定义方式。

```
const int * num;
int const * num;
```

以下两点需要注意：

1. 常量指针说的是不能通过这个指针改变变量的值，但可以通过其他的引用来改变变量的值。

```
int cnt=5;
const int* num=&cnt;
cnt=6;
```

1. 常量指针指向的值不能改变，但这并不意味着指针本身不能改变，常量指针可以指向其他的地址。

```
int cnt=5;
int tmp=6;
const int* num=&cnt;
num=&tmp;
```

#### 指针常量

指针常量是指指针本身是个常量，不能再指向其他的地址，写法如下：

```
int *const num;
```

需要注意的是，指针常量指向的地址不能改变，但是地址中保存的数值是可以改变的，可以通过其他指向改地址的指针来修改。

```
int cnt=5;
int *tmp=&cnt;
int* const num=&cnt;
*tmp=6;
```

区分常量指针和指针常量的关键就在于`星号的位置`，我们以星号为分界线。

- 如果const在星号的左边，则为常量指针
- 如果const在星号的右边则为指针常量

如果我们将星号读作‘指针’，将const读作‘常量’的话，内容正好符合。

- `int const * num；`是常量指针，
- `int *const num；`是指针常量。

#### 指向常量的常指针

还有一种情况是指向常量的常指针，这相当于是常量指针与指针常量的结合，指针指向的位置不能改变并且也不能通过这个指针改变`变量的值`，例如

```
const int* const num;
```

这个代表num所指向的对象的值以及它的地址本身都不能被改变

#### 修饰函数的形参



根据常量指针与指针常量，const修饰函数的参数也是分为三种情况

1. 防止修改指针指向的内容

```
void FUN(char *destin, const char *source);
```

其中 source 是输入参数，destin 是输出参数。给 source 加上 const 修饰后，如果函数体内的语句试图改动 source 的内容，编译器将报错，但反过来是可以的，编译器允许将`char *`类型的数据赋值给`const char *`类型的变量。

1. 防止修改指针指向的地址

```
void FUN ( int * const p1 , int * const p2 )
```

指针p1和指针p2指向的地址都不能修改。

1. 以上两种的结合。

在C语言标准库中，有很多函数的形参都被 const 限制了，下面是部分函数的原型：

```
size_t strlen ( const char * str );
int strcmp ( const char * str1, const char * str2 );
char * strcat ( char * destination, const char * source );
char * strcpy ( char * destination, const char * source );
int system (const char* command);
int puts ( const char * str );
int printf ( const char * format, ... );
```

#### 修饰函数的返回值



如果给以“`指针传递`”方式的函数返回值加 const 修饰，那么函数返回值（即指针）的内容不能被修改，该返回值只能被赋给加const 修饰的同类型指针，例如

```
const char * FUN(void);
```

如下语句将出现编译错误：

```
char *str = FUN();
```

正确的用法是

```
const char *str = FUN();
```

#### 思考

- C与C++中的const用法有什么区别？
- 编译器会给const定义的变量分配存储空间吗？
- const变量能被其他文件extern引用吗？

参考：C语言中文网





### static





### inline





### cast























