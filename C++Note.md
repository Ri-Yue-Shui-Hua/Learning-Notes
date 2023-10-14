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





## std::function介绍

类模版`std::function`是一种通用、多态的函数封装。`std::function`的实例可以对任何可以调用的目标实体进行存储、复制、和调用操作，这些目标实体包括普通函数、Lambda表达式、函数指针、以及其它函数对象等。`std::function`对象是对C++中现有的可调用实体的一种类型安全的包裹（我们知道像函数指针这类可调用实体，是类型不安全的）。

通常std::function是一个函数对象类，它包装其它任意的函数对象，被包装的函数对象具有类型为T1, …,TN的N个参数，并且返回一个可转换到R类型的值。`std::function`使用 模板转换构造函数接收被包装的函数对象；特别是，闭包类型可以隐式地转换为`std::function`。

最简单的理解就是：

　　　通过`std::function`对C++中各种可调用实体（普通函数、Lambda表达式、函数指针、以及其它函数对象等）的封装，形成一个新的可调用的`std::function`对象；让我们不再纠结那么多的可调用实体。一切变的简单粗暴。



```cpp
#include <functional>
#include <iostream>
using namespace std;

std::function< int(int)> Functional;

// 普通函数
int TestFunc(int a)
{
	return a;
}

// Lambda表达式
auto lambda = [](int a)->int{ return a; };

// 仿函数(functor)
class Functor
{
public:
	int operator()(int a)
	{
		return a;
	}
};

// 1.类成员函数
// 2.类静态函数
class TestClass
{
public:
	int ClassMember(int a) { return a; }
	static int StaticMember(int a) { return a; }
};

int main()
{
	// 普通函数
	Functional = TestFunc;
	int result = Functional(10);
	cout << "普通函数："<< result << endl;
	
	// Lambda表达式
	Functional = lambda;
	result = Functional(20);
	cout << "Lambda表达式："<< result << endl;
	
	// 仿函数
	Functor testFunctor;
	Functional = testFunctor;
	result = Functional(30);
	cout << "仿函数："<< result << endl;
	
	// 类成员函数
	TestClass testObj;
	Functional = std::bind(&TestClass::ClassMember, testObj, std::placeholders::_1);
	result = Functional(40);
	cout << "类成员函数："<< result << endl;
	
	// 类静态函数
	Functional = TestClass::StaticMember;
	result = Functional(50);
	cout << "类静态函数："<< result << endl;
	
	return 0;
}
```



result:

```bash
普通函数：10
Lambda表达式：20
仿函数：30
类成员函数：40
类静态函数：50

--------------------------------
Process exited after 0.07694 seconds with return value 0

Press ANY key to exit...
```



## std::bind

同时用到了：` std::placeholders`.

### 概述



`bind`函数可以看作一个通用的函数适配器，所谓适配器，即使某种事物的行为类似于另外一种事物的一种机制，如容器适配器：`stack(栈)`、`queue(队列)`、`priority_queue(优先级队列)`。
 `bind`函数接受一个可调用对象，生成一个新的可调用对象来适配原对象。

### 函数原型

```kotlin
template <class Fn, class... Args>
  /* unspecified */ bind (Fn&& fn, Args&&... args);
```

`bind`函数接受一个逗号分割的参数列表`args`，对应给定函数对象`fn`的参数，返回一个新的函数对象。
 参数列表`args`中：

- 如果绑定到一个值，则调用返回的函数对象将始终使用该值作为参数
- 如果是一个形如`_n`的占位符，则调用返回的函数对象会转发传递给调用的参数(该参数的顺序号由占位符指定)

### 使用



```cpp
#include <functional> //bind函数 placeholders命名空间

int Plus(int x, int y) {
    return x + y;
}

int PlusOne(int x) {
    auto func = std::bind(Plus, std::placeholders::_1, 1);
    return func(x);
}

int main()
{
    std::cout << PlusOne(9) << std::endl; //结果 10
    return 0;
}
```

**注：**示例代码中的`_1`即为形如`_n`的占位符，其定义在命名空间`placeholders`中，而此命名空间又定义在命名空间`std`中

### 使用场景

根据`bind`函数的特征，有以下几个场景时可以使用`bind`：

1. 当`bind`函数的参数列表绑定到一个值时，则调用返回的函数对象将始终使用该值作为参数。所以`bind`函数可以将一个函数的参数特例化，如上文的示例代码
2. 当`bind`函数的参数列表是一个占位符时，调用返回的函数对象的参数顺序号由占位符指定，所以`bind`函数可以对调用函数对象的参数重新安排顺序，例如：



```swift
using namespace std;

void output(int a, int b, int c) {
    cout << a << " " << b << " " << c;
}

int main()
{
    auto func = bind(output, placeholders::_2, placeholders::_1, placeholders::_3);
    func(1, 2, 3); //结果 2 1 3
    return 0;
}
```

1. 与`std::function`配合，实现回调函数。具体见文章[C++ std::function](https://www.jianshu.com/p/4ea00ee0dabd)，这里不再赘述。

### 其他

一个知识点厉不厉害，归根到底还是要经过实践的考验，下面就来看看`std::bind`到底怎么用。

先看看《[C++11中的std::function](https://blog.csdn.net/u013654125/article/details/100140547)》中那段代码，`std::function`可以绑定全局函数，静态函数，但是绑定类的成员函数时，必须要借助`std::bind`的帮忙。但是话又说回来，不借助`std::bind`也是可以完成的，只需要传一个*this变量进去就好了，比如：

```cpp
#include <iostream>
#include <functional>
using namespace std;

class View
{
public:
	void onClick(int x, int y)
	{
		cout << "X : " << x << ", Y : " << y << endl;
	}
};

// 定义function类型, 三个参数
function<void(View, int, int)> clickCallback;

int main(int argc, const char * argv[])
{
	View button;
	
	// 指向成员函数
	clickCallback = &View::onClick;
	
	// 进行调用
	clickCallback(button, 10, 123);
	return 0;
}
```

result:

```bash
X : 10, Y : 123

--------------------------------
Process exited after 0.008874 seconds with return value 0

Press ANY key to exit...
```

再来一段示例谈谈怎么使用std::bind代码：

```cpp
#include <iostream>
#include <functional>
using namespace std;

int TestFunc(int a, char c, float f)
{
	cout << a << endl;
	cout << c << endl;
	cout << f << endl;
	
	return a;
}

int main()
{
	auto bindFunc1 = bind(TestFunc, std::placeholders::_1, 'A', 100.1);
	bindFunc1(10);
	
	cout << "=================================\n";
	
	auto bindFunc2 = bind(TestFunc, std::placeholders::_2, std::placeholders::_1, 100.1);
	bindFunc2('B', 10);
	
	cout << "=================================\n";
	
	auto bindFunc3 = bind(TestFunc, std::placeholders::_2, std::placeholders::_3, std::placeholders::_1);
	bindFunc3(100.1, 30, 'C');
	
	return 0;
}
```

result:

```bash
10
A
100.1
=================================
10
B
100.1
=================================
30
C
100.1

--------------------------------
Process exited after 0.02071 seconds with return value 0

Press ANY key to exit...
```

以下是使用std::bind的一些需要注意的地方：

- bind预先绑定的参数需要传具体的变量或值进去，对于预先绑定的参数，是pass-by-value的；
- 对于不事先绑定的参数，需要传std::placeholders进去，从_1开始，依次递增。placeholder是pass-by-reference的；
- bind的返回值是可调用实体，可以直接赋给std::function对象；
- 对于绑定的指针、引用类型的参数，使用者需要保证在可调用实体调用之前，这些参数是可用的；
- 类的this可以通过对象或者指针来绑定。



## std::lambda



lambda是C++11中才引入的新特性，能定义匿名对象，而不必定义独立的函数和函数对象。

在介绍函数对象的for_each例子中，如果不用创建函数对象，可以使用下面

std::for_each(dest.begin(), dest.end(), [](int i){ std::cout << ' ' << i; });

上述代码中红色部分就是ambda表达式，编译器会对这部分代码生成一个匿名的函数对象类。

如果只是在某一处使用，使用lambda表示更加简洁，不用特意写一个函数或者函数对象类；使用lambda表达式表达能力更强，提高代码清晰度。

 

### lambda的4种不同形式

[ capture ] ( params ) mutable exception attribute -> ret { body }  —— 这是一个完整的声明。

[ capture ] ( params ) -> ret { body }  ——去掉了mutable关键字，即不能修改捕获外部对象的值（外部变量在capture中定义，见后面介绍）

[ capture ] ( params ) { body } ——去掉了返回值类型ret的定义，要么根据函数体body中的return自动推倒，要么返回类型是void（上述例子用的就是这个形式）。

[ capture ] { body } ——去掉了输入参数列表params的定义，即函数参数列表空，是()

 

### [ capture ]说明

该部分指定了哪些外部变量可以在lambda函数体body中可见,符号可按如下规则传入:

1. []      不捕获任何外部变量 
2. [=]     以值的形式捕获lambda表达式所在函数的函数体中的所有外部变量 
3. [&]     以引用的形式捕获lambda表达式所在函数的函数体中的所有外部变量
4. [a,&b]  按值捕获a，并按引用捕获b 
5. [=, &a] 以引用的形式捕获a，其余变量以值的形式捕获 
6. [&， a] 以值的形式捕获a，其余变量以引用的形式捕获 
7. [this]  按值捕获了this指针 

### 其他参考

[C++11 lambda表达式精讲 (biancheng.net)](http://c.biancheng.net/view/3741.html)

```cpp
class A
{
    public:
    int i_ = 0;
    void func(int x, int y)
    {
        auto x1 = []{ return i_; };                    // error，没有捕获外部变量
        auto x2 = [=]{ return i_ + x + y; };           // OK，捕获所有外部变量
        auto x3 = [&]{ return i_ + x + y; };           // OK，捕获所有外部变量
        auto x4 = [this]{ return i_; };                // OK，捕获this指针
        auto x5 = [this]{ return i_ + x + y; };        // error，没有捕获x、y
        auto x6 = [this, x, y]{ return i_ + x + y; };  // OK，捕获this指针、x、y
        auto x7 = [this]{ return i_++; };              // OK，捕获this指针，并修改成员的值
    }
};
int a = 0, b = 1;
auto f1 = []{ return a; };               // error，没有捕获外部变量
auto f2 = [&]{ return a++; };            // OK，捕获所有外部变量，并对a执行自加运算
auto f3 = [=]{ return a; };              // OK，捕获所有外部变量，并返回a
auto f4 = [=]{ return a++; };            // error，a是以复制方式捕获的，无法修改
auto f5 = [a]{ return a + b; };          // error，没有捕获变量b
auto f6 = [a, &b]{ return a + (b++); };  // OK，捕获a和b的引用，并对b做自加运算
auto f7 = [=, &b]{ return a + (b++); };  // OK，捕获所有外部变量和b的引用，并对b做自加运算
```



## std::min_element

[C++ STL min_element 使用说明 - 简书 (jianshu.com)](https://www.jianshu.com/p/959ac770fb17)

`std::min_element` 用于寻找范围 [first, last) 中的最小元素。

前2个参数指定容器的范围，第3个参数是比较函数，为可选参数。
 返回值为指向范围 [first, last) 中最小元素的迭代器。
 若范围中有多个元素等价于最小元素，则返回指向首个这种元素的迭代器。若范围为空则返回 last 。

关于比较函数， 默认是用 operator< 比较元素， 也可以自定义比较函数。
 所以`std::min_element`两种函数签名如下：

```cpp
template< class ForwardIt > 
constexpr ForwardIt min_element( ForwardIt first, ForwardIt last );
```



```cpp
template< class ForwardIt, class Compare >
constexpr ForwardIt min_element( ForwardIt first, ForwardIt last, Compare comp );
```

`std::max_element`与`std::min_element`类似，只是用于寻找最大的元素。

### 头文件

```
#include <algorithm>
```

### 例子：求数组里最下的元素

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main(int argc, char **argv) 
{  
    std::vector<int> v{3, 1, 4, 1, 5, 9};
    
    std::vector<int>::iterator minElement = std::min_element(v.begin(), v.end());
    
    std::cout << "min element: " << *(minElement) << std::endl;
    std::cout << "min element at:" << std::distance(v.begin(), minElement) << std::endl;
    return 0;
}
```

result:

```bash
min element: 1
min element at:1

--------------------------------
Process exited after 0.03599 seconds with return value 0

Press ANY key to exit...
```

### 例子：自定义比较函数

比如如下自定义比较函数，把求最小指编程求最大值

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main(int argc, char **argv) 
{  
    std::vector<int> v{3, 1, 4, 1, 5, 9};
    
    auto comp = [](int i, int j){ return i>j;}; //  i<j : min;  i>j : max   
    
    std::vector<int>::iterator maxElement = std::min_element(v.begin(), v.end(), comp);
    
    std::cout << "max element: " << *(maxElement) << std::endl;
    std::cout << "max element at:" << std::distance(v.begin(), maxElement) << std::endl;
    return 0;
}
```

result:

```bash
max element: 9
max element at:5

--------------------------------
Process exited after 0.0654 seconds with return value 0

Press ANY key to exit...
```

### std::min_element 与 std::min 的区别



`std::min`一般用于求 a 与 b 的较小者 或者求 initializer_list ilist 中值的最小者。
 `std::min_element`是求一个范围内的最小者的迭代器。范围可以是全部容器，也可以是容器的一个子区间。
 所以它们的适用范围和返回值不一样。

### 参考

[https://zh.cppreference.com/w/cpp/algorithm/min_element](https://links.jianshu.com/go?to=https%3A%2F%2Fzh.cppreference.com%2Fw%2Fcpp%2Falgorithm%2Fmin_element)
 [http://www.cplusplus.com/reference/algorithm/min_element/](https://links.jianshu.com/go?to=http%3A%2F%2Fwww.cplusplus.com%2Freference%2Falgorithm%2Fmin_element%2F)
 [https://zh.cppreference.com/w/cpp/algorithm/min](https://links.jianshu.com/go?to=https%3A%2F%2Fzh.cppreference.com%2Fw%2Fcpp%2Falgorithm%2Fmin)





## #program 



参考：https://www.cnblogs.com/runningRain/p/5936788.html

C++ #pragma 预处理指令

#pragma 预编译指令的作用是设定编译器的状态或者是指示编译器完成一些特定的动作。#pragma指令对每个编译器给出了一个方法，在保持与C和C++语言完全兼容的情况下，给出主机或操作系统专有的特征。

　　其使用的格式一般为: `#pragma Para`。其中Para 为参数，常见的参数如下：

### Message参数

　　Message参数编译信息输出窗口中输出相应地信息，使用方法如下：

```cpp
#pragma message("消息文本")
```

使用示例，假如在程序中我们定义了很多宏来控制源代码版本的时候，我们自己有可能都会忘记有没有正确的设置这些宏，此时我们可以用这条指令在编译的时候就进行检查。假设我们希望判断自己有没有在源代码的什么地方定义了_X86这个宏可以用下面的方法：

```cpp
#ifdef _X86
#pragma message("_X86 macro activated!")
#endif
```



### code_seg参数

　　code_seg参数可以设置程序中函数代码存放的代码段，使用方式如下：

```cpp
#pragma code_seg(["section-name"[,"section-class"]])
```

### \#program once 参数

　　其作用是在在头文件的最开始加入这条指令，以保证头文件被编译一次。但#program once是编译器相关的，就是说在这个编译系统上能用，但在其他的编译系统上就不一定能用，所以其可移植性较差。一般如果强调程序跨平台，还是选择使用`“#ifndef,  #define,  #endif”`比较好。

### \#program hdrstop

　　#program hdrstop表示预编译头文件到此为止，后面的头文件不进行编译。

### \#program resource 

　　#program resource  “*.dfm”表示把*.dfm文件中的资源添加到工程。

### \#program comment

　　#program comment将一个注释记录放入一个对象文件或可执行文件。

### \#program data_seg

　　#program data_seg用来建立一个新的数据段并定义共享数据。如下：

```cpp
#pragma data_seg（"shareddata")
HWNDsharedwnd=NULL;//共享数据
#pragma data_seg()
```

**说明：a.** #pragma data_seg()一般用于DLL中。也就是说，在DLL中定义一个共享的有名字的数据段。最关键的是：这个数据段中的全局变量可以被多个进程共享,否则多个进程之间无法共享DLL中的全局变量。

　　　**b.** 共享数据必须初始化，否则微软编译器会把没有初始化的数据放到.BSS段中，从而导致多个进程之间的共享行为失败。例如:

```cpp
#pragma data_seg("MyData")
intg_Value;　　　　//Note that the global is not initialized.
#pragma data_seg()
//DLL提供两个接口函数：
int GetValue()
{
    return g_Value;
}
void SetValue(int n)
{
    g_Value=n;
}
```

**解释：**启动两个进程A和B，A和B都调用了这个DLL，假如A调用了SetValue(5); B接着调用int m = GetValue(); 那么m的值不一定是5，而是一个未定义的值。因为DLL中的全局数据对于每一个调用它的进程而言，是私有的，不能共享的。假如你对g_Value进行了初始化，那么g_Value就一定会被放进MyData段中。换句话说，如果A调用了SetValue(5); B接着调用int m = GetValue(); 那么m的值就一定是5，这就实现了跨进程之间的数据通信。

### \#program region

　　#program region用于折叠特定的代码段，示例如下：

```cpp
#pragma region Variables
HWND hWnd;
const size_t Max_Length = 20;
//other variables
#pragma endregion This region contains global variables.
```







## 获取时间

### time



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

### chrono

#### 时间转换



```cpp
#include <iostream>
#include <string>
#include <chrono>


int main()
{
	std::chrono::hours hour_time = std::chrono::hours(1);
	
	std::chrono::minutes minutes_time = std::chrono::duration_cast<std::chrono::minutes>(hour_time);
	
	std::chrono::seconds seconds_time = std::chrono::duration_cast<std::chrono::seconds>(hour_time);
	
	std::chrono::milliseconds milliseconds_time = std::chrono::duration_cast<std::chrono::milliseconds>(hour_time);
	
	std::chrono::microseconds microseconds_time = std::chrono::duration_cast<std::chrono::microseconds>(hour_time);
	
	std::cout << "1小时可转换为 \n"
	<< minutes_time.count() << "分钟 \n"
	<< seconds_time.count() << "秒 \n"
	<< milliseconds_time.count() << "毫秒 \n"
	<< microseconds_time.count() << "微秒" << std::endl;
	
	getchar();
	return 0;
}
```

result:

```bash
1小时可转换为
60分钟
3600秒
3600000毫秒
3600000000微秒
```



#### 输出时刻

```cpp
#include <iostream>
#include <string>
#include <chrono>


int main()
{
	auto t1 = std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::system_clock::now().time_since_epoch())
	.count();
	std::cout << "t1 : \n" << t1 << std::endl;
	auto t2 = std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::system_clock::now().time_since_epoch())
	.count();
	std::cout << "t2 : \n" << t2 << std::endl;
	getchar();
	return 0;
}
```

result:

```bash
t1 :
1663575818607
t2 :
1663575818608
```

#### 输出时间



```cpp
#include <iostream>
#include <chrono>
#include <ratio>
#include <thread>

void f()
{
	std::this_thread::sleep_for(std::chrono::seconds(1));
}

int main()
{
	auto t1 = std::chrono::high_resolution_clock::now();
	f();
	auto t2 = std::chrono::high_resolution_clock::now();
	
	// integral duration: requires duration_cast
	auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
	
	// fractional duration: no duration_cast needed
	std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
	
	std::cout << "f() took " << fp_ms.count() << " ms, "
	<< "or " << int_ms.count() << " whole milliseconds\n";
}
```

result:

```bash
f() took 1003.15 ms, or 1003 whole milliseconds

--------------------------------
Process exited after 1.071 seconds with return value 0

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

const 关键字不能与 static 关键字同时使用，因为 static 关键字修饰静态成员函数，静态成员函数不含有 this 指针，即不能实例化，const 成员函数必须具体到某一实例。

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

[C/C++ 中 static 的用法全局变量与局部变量 | 菜鸟教程 (runoob.com)](https://www.runoob.com/w3cnote/cpp-static-usage.html)

#### 什么是static?



static 是 C/C++ 中很常用的修饰符，它被用来控制变量的存储方式和可见性。

#####  static 的引入

我们知道在函数内部定义的变量，当程序执行到它的定义处时，编译器为它在栈上分配空间，**函数在栈上分配的空间在此函数执行结束时会释放掉**，这样就产生了一个问题: **如果想将函数中此变量的值保存至下一次调用时**，如何实现？ 最容易想到的方法是定义为全局的变量，但定义一个全局变量有许多缺点，最明显的缺点是破坏了此变量的访问范围（使得在此函数中定义的变量，不仅仅只受此函数控制）。**static 关键字则可以很好的解决这个问题**。

另外，在 C++ 中，需要一个数据对象为整个类而非某个对象服务,同时又力求不破坏类的封装性,即要求此成员隐藏在类的内部，对外不可见时，可将其定义为静态数据。

##### 静态数据的存储

**全局（静态）存储区**：分为 DATA 段和 BSS 段。DATA 段（全局初始化区）存放初始化的全局变量和静态变量；BSS 段（全局未初始化区）存放未初始化的全局变量和静态变量。程序运行结束时自动释放。其中BBS段在程序执行之前会被系统自动清0，所以未初始化的全局变量和静态变量在程序执行之前已经为0。存储在静态数据区的变量会在程序刚开始运行时就完成初始化，也是唯一的一次初始化。

在 C++ 中 static 的内部实现机制：静态数据成员要在程序一开始运行时就必须存在。因为函数在程序运行中被调用，所以静态数据成员不能在任何函数内分配空间和初始化。

这样，**它的空间分配有三个可能的地方**，一是作为类的外部接口的头文件，那里有类声明；二是类定义的内部实现，那里有类的成员函数定义；三是应用程序的 main() 函数前的全局数据声明和定义处。

静态数据成员要实际地分配空间，故不能在类的声明中定义（只能声明数据成员）。类声明只声明一个类的"尺寸和规格"，并不进行实际的内存分配，所以在类声明中写成定义是错误的。它也不能在头文件中类声明的外部定义，因为那会造成在多个使用该类的源文件中，对其重复定义。

**static** 被引入以告知编译器，**将变量存储在程序的静态存储区而非栈上空间，静态数据成员按定义出现的先后顺序依次初始化，注意静态成员嵌套时，要保证所嵌套的成员已经初始化了。消除时的顺序是初始化的反顺序**。

**优势：**可以节省内存，因为它是所有对象所公有的，因此，对多个对象来说，静态数据成员只存储一处，供所有对象共用。静态数据成员的值对每个对象都是一样，但它的值是可以更新的。只要对静态数据成员的值更新一次，保证所有对象存取更新后的相同的值，这样可以提高时间效率。

#### 在 C/C++ 中static的作用

##### 总的来说

- （1）在修饰变量的时候，static 修饰的静态局部变量**只执行初始化一次**，而且延长了局部变量的生命周期，直到程序运行结束以后才释放。
- （2）static **修饰全局变量**的时候，这个全局变量**只能在本文件中访问**，不能在其它文件中访问，即便是 extern 外部声明也不可以。
- （3）static **修饰一个函数**，则这个函数的**只能在本文件中调用**，不能被其他文件调用。static 修饰的变量存放在全局数据区的静态变量区，包括全局静态变量和局部静态变量，都在全局数据区分配内存。初始化的时候自动初始化为 0。
- （4）不想被释放的时候，可以使用static修饰。比如修饰函数中存放在栈空间的数组。如果不想让这个数组在函数调用结束释放可以使用 static 修饰。
- （5）考虑到数据安全性（当程序想要使用全局变量的时候应该先考虑使用 static）。

##### 静态变量与普通变量

**静态全局变量有以下特点：**

- （1）静态变量都在全局数据区分配内存，包括后面将要提到的静态局部变量;
- （2）未经初始化的静态全局变量会被程序自动初始化为0（在函数体内声明的自动变量的值是随机的，除非它被显式初始化，而在函数体外被声明的自动变量也会被初始化为 0）；
- （3）静态全局变量在声明它的整个文件都是可见的，而在文件之外是不可见的。

**优点：**静态全局变量不能被其它文件所用；其它文件中可以定义相同名字的变量，不会发生冲突。

**（1）全局变量和全局静态变量的区别**

- 1）全局变量是不显式用 static 修饰的全局变量，全局变量默认是有外部链接性的，作用域是整个工程，在一个文件内定义的全局变量，在另一个文件中，通过 extern 全局变量名的声明，就可以使用全局变量。
- 2）全局静态变量是显式用 static 修饰的全局变量，作用域是声明此变量所在的文件，其他的文件即使用 extern 声明也不能使用。

##### 静态局部变量有以下特点：

- （1）该变量在全局数据区分配内存；
- （2）静态局部变量在程序执行到该对象的声明处时被首次初始化，即以后的函数调用不再进行初始化；
- （3）静态局部变量一般在声明处初始化，如果没有显式初始化，会被程序自动初始化为 0；
- （4）它始终驻留在全局数据区，直到程序运行结束。但其作用域为局部作用域，当定义它的函数或语句块结束时，其作用域随之结束。

一般程序把新产生的动态数据存放在堆区，函数内部的自动变量存放在栈区。自动变量一般会随着函数的退出而释放空间，静态数据（即使是函数内部的静态局部变量）也存放在全局数据区。全局数据区的数据并不会因为函数的退出而释放空间。

例子

```cpp
//example:
#include <stdio.h>  
#include <stdlib.h>  
int k1 = 1;
int k2;
static int k3 = 2;
static int k4;
int main()
{
    static int m1 = 2, m2;
    int i = 1;
    char*p;
    char str[10] = "hello";
    char*q = "hello";
    p = (char *)malloc(100);
    free(p);
    printf("栈区-变量地址    i：%p\n", &i);
    printf("栈区-变量地址   p：%p\n", &p);
    printf("栈区-变量地址 str：%p\n", str);
    printf("栈区-变量地址   q：%p\n", &q);
    printf("堆区地址-动态申请：%p\n", p);
    printf("全局外部有初值 k1：%p\n", &k1);
    printf("   外部无初值 k2：%p\n", &k2);
    printf("静态外部有初值 k3：%p\n", &k3);
    printf("   外静无初值 k4：%p\n", &k4);
    printf("  内静态有初值 m1：%p\n", &m1);
    printf("  内静态无初值 m2：%p\n", &m2);
    printf("    文字常量地址：%p, %s\n", q, q);
    printf("      程序区地址：%p\n", &main);
    return 0;
}
```

运行输出：

![image-20220830203829573](C++Note.assets/image-20220830203829573.png)

####  static 用法

##### 在 C++ 中

static 关键字最基本的用法是：

- 1、被 static 修饰的变量属于类变量，可以通过**类名.变量名**直接引用，而不需要 new 出一个类来
- 2、被 static 修饰的方法属于类方法，可以通过**类名.方法名**直接引用，而不需要 new 出一个类来

被 static 修饰的变量、被 static 修饰的方法统一属于类的静态资源，是类实例之间共享的，换言之，一处变、处处变。

在 C++ 中，静态成员是属于整个类的而不是某个对象，静态成员变量只存储一份供所有对象共用。所以在所有对象中都可以共享它。使用静态成员变量实现多个对象之间的数据共享不会破坏隐藏的原则，保证了安全性还可以节省内存。

静态成员的定义或声明要加个关键 static。静态成员可以通过双冒号来使用即 **<类名>::<静态成员名>**。

##### 静态类相关

通过类名调用静态成员函数和非静态成员函数:

```cpp
class Point  
{  
public:   
    void init()  
    {    
    }  
    static void output()  
    {  
    }  
};  
void main()  
{  
    Point::init();  
    Point::output();  
}
```



![image-20220830204333430](C++Note.assets/image-20220830204333430.png)

报错：

```bash
'Point::init' : illegal call of non-static member function
```

**结论 1：**不能通过类名来调用类的非静态成员函数。

通过类的对象调用静态成员函数和非静态成员函数。

```cpp
//example:
#include <stdio.h>  
#include <stdlib.h>  
class Point  
{  
	public:   
	void init()  
	{    
	}  
	static void output()  
	{  
	}  
}; 
int main()  
{  
	Point pt;  
	pt.init();  
	pt.output(); 
	return 0;
}
```

编译通过。

**结论 2**：类的对象可以使用静态成员函数和非静态成员函数。

在类的静态成员函数中使用类的非静态成员。

```cpp
#include <stdio.h>  
class Point  
{  
	public:   
	void init()  
	{    
	}  
	static void output()  
	{  
		printf("%d\n", m_x);  
	}  
	private:  
	int m_x;  
};  
int main()  
{  
	Point pt;  
	pt.output();  
	return 0;
}
```



![image-20220830205115202](C++Note.assets/image-20220830205115202.png)

编译出错：

```bash
 error: invalid use of member 'Point::m_x' in static member function
```

因为静态成员函数属于整个类，在类实例化对象之前就已经分配空间了，而类的非静态成员必须在类实例化对象后才有内存空间，所以这个调用就出错了，就好比没有声明一个变量却提前使用它一样。

**结论3：**静态成员函数中不能引用非静态成员。

在类的非静态成员函数中使用类的静态成员。

```cpp
#include <stdio.h>  
class Point  
{  
	public:   
	void init()  
	{    
		output();  
	}  
	static void output()  
	{  
	}  
};  
int main()  
{  
	Point pt;  
	pt.init();
	pt.output(); 
	return 0;
}
```

编译通过。

**结论 4**：类的非静态成员函数可以调用用静态成员函数，但反之不能。

使用类的静态成员变量。

```cpp
#include <stdio.h>  
class Point  
{  
	public:   
	Point()  
	{    
		m_nPointCount++;  
	}  
	~Point()  
	{  
		m_nPointCount--;  
	}  
	static void output()  
	{  
		printf("%d\n", m_nPointCount);  
	}  
	private:  
	static int m_nPointCount;  
};  
int main()  
{  
	Point pt;  
	pt.output(); 
	return 0;
}
```

编译无错误， 生成 EXE 程序时报链接错误。

```bash
86_64-w64-mingw32/bin/ld.exe: main.o:main.cpp:(.rdata$.refptr._ZN5Point13m_nPointCountE[.refptr._ZN5Point13m_nPointCountE]+0x0): undefined reference to `Point::m_nPointCount'
collect2.exe: error: ld returned 1 exit status
```

这是因为类的静态成员变量在使用前必须先初始化。

在 **main()** 函数前加上 **int Point::m_nPointCount = 0;** 再编译链接无错误，运行程序将输出 1。

**结论 5**：类的静态成员变量必须先初始化再使用。

**思考总结：**静态资源属于类，但是是独立于类存在的。从 J 类的加载机制的角度讲，静态资源是类初始化的时候加载的，而非静态资源是类实例化对象的时候加载的。 类的初始化早于类实例化对象，比如 **Class.forName("xxx")** 方法，就是初始化了一个类，但是并没有实例化对象，只是加载这个类的静态资源罢 了。所以对于静态资源来说，它是不可能知道一个类中有哪些非静态资源的；但是对于非静态资源来说就不一样了，由于它是实例化对象出来之后产生的，因此属于类的这些东西它都能认识。所以上面的几个问题答案就很明确了：

- 1）静态方法能不能引用非静态资源？不能，实例化对象的时候才会产生的东西，对于初始化后就存在的静态资源来说，根本不认识它。
- 2）静态方法里面能不能引用静态资源？可以，因为都是类初始化的时候加载的，大家相互都认识。
- 3）非静态方法里面能不能引用静态资源？可以，非静态方法就是实例方法，那是实例化对象之后才产生的，那么属于类的内容它都认识。

（**static 修饰类：**这个用得相对比前面的用法少多了，static 一般情况下来说是不可以修饰类的， 如果 static 要修饰一个类，说明这个类是一个静态内部类（注意 static 只能修饰一个内部类），也就是匿名内部类。像线程池 ThreadPoolExecutor 中的四种拒绝机制 CallerRunsPolicy、AbortPolicy、DiscardPolicy、 DiscardOldestPolicy 就是静态内部类。静态内部类相关内容会在写内部类的时候专门讲到。）

#### 总结：



- （1）静态成员函数中不能调用非静态成员。
- （2）非静态成员函数中可以调用静态成员。因为静态成员属于类本身，在类的对象产生之前就已经存在了，所以在非静态成员函数中是可以调用静态成员的。
- （3）静态成员变量使用前必须先初始化(如 **int MyClass::m_nNumber = 0;**)，否则会在 linker 时出错。

**一般总结**：在类中，static 可以用来修饰静态数据成员和静态成员方法。

**静态数据成员**

- （1）静态数据成员可以实现多个对象之间的数据共享，它是类的所有对象的共享成员，它在内存中只占一份空间，如果改变它的值，则各对象中这个数据成员的值都被改变。
- （2）静态数据成员是在程序开始运行时被分配空间，到程序结束之后才释放，只要类中指定了静态数据成员，即使不定义对象，也会为静态数据成员分配空间。
- （3）静态数据成员可以被初始化，但是只能在类体外进行初始化，若未对静态数据成员赋初值，则编译器会自动为其初始化为 0。
- （4）静态数据成员既可以通过对象名引用，也可以通过类名引用。

**静态成员函数**

- （1）静态成员函数和静态数据成员一样，他们都属于类的静态成员，而不是对象成员。
- （2）非静态成员函数有 this 指针，而静态成员函数没有 this 指针。
- （3）静态成员函数主要用来访问静态数据成员而不能访问非静态成员。

再给一个利用类的静态成员变量和函数的例子以加深理解，这个例子建立一个学生类，每个学生类的对象将组成一个双向链表，用一个静态成员变量记录这个双向链表的表头，一个静态成员函数输出这个双向链表。

```cpp
#include <stdio.h>  
#include <string.h>
const int MAX_NAME_SIZE = 30;  

class Student  
{  
	public:  
	Student(char *pszName);
	~Student();
public:
	static void PrintfAllStudents();
	private:  
	char    m_name[MAX_NAME_SIZE];  
	Student *next;
	Student *prev;
	static Student *m_head;
};  

Student::Student(char *pszName)
{  
	strcpy(this->m_name, pszName);
	
	//建立双向链表，新数据从链表头部插入。
	this->next = m_head;
	this->prev = NULL;
	if (m_head != NULL)
		m_head->prev = this;
	m_head = this;  
}  

Student::~Student ()//析构过程就是节点的脱离过程  
{  
	if (this == m_head) //该节点就是头节点。
	{
		m_head = this->next;
	}
	else
	{
		this->prev->next = this->next;
		this->next->prev = this->prev;
	}
}  

void Student::PrintfAllStudents()
{
	for (Student *p = m_head; p != NULL; p = p->next)
		printf("%s\n", p->m_name);
}

Student* Student::m_head = NULL;  

int main()  
{   
	Student studentA("AAA");
	Student studentB("BBB");
	Student studentC("CCC");
	Student studentD("DDD");
	Student student("MoreWindows");
	Student::PrintfAllStudents();
	return 0;
}
```

程序将输出:

![image-20220830210149925](C++Note.assets/image-20220830210149925.png)



[(7条消息) C++ static详解_鬼筠的博客-CSDN博客_c++static](https://blog.csdn.net/u010797208/article/details/41549461)



静态局部变量有以下特点:
01.其内存存放在 程序的全局数据区中,
02.静态局部变量在程序执行到该对象声明时,会被首次初始化.其后运行到该对象的声明时,不会再次初始化,这也是为什么上面程序测试函数每次输出的值都是递增的原因.
03.如果静态局部变量没有被显示初始化,则其值会自动被系统初始化为0.
04.局部静态变量 不能被其作用域之外的其他模块调用,其调用范围仅限于声明该变量的函数作用域当中.



在函数的返回类型前加上关键字static,可以将此函数声明为静态函数.静态函数与普通函数不同,其作用域只在声明它的文件当中.其他文件可以定义同名的全局函数,而不冲突.
想要在其他文件调用静态函数 需要显示的调用extern关键字修饰其声明.否则编译器会link error.

总结定义静态函数的好处:

01.其他文件可以定义同名函数
02.静态函数不会被其他文件所引用,其作用域只在当前声明他的文件中.



静态数据成员的特点:

01.静态数据成员的服务对象并非是单个类实例化的对象,而是所有类实例化的对象(这点可以用于设计模式中的单例模式实现).
02.静态数据成员必须显示的初始化分配内存,在其包含类没有任何实例花之前,其已经有内存分配.
03.静态数据成员与其他成员一样,遵从public,protected,private的访问规则.
04.静态数据成员内存存储在全局数据区,只随着进程的消亡而消亡.

静态数据成员与全局变量相比的优势:
01.静态数据成员不进入程序全局名字空间,不会与其他全局名称的同名同类型变量冲突.

静态局部函数的特点如下:

01.静态成员函数比普通成员函数多了一种调用方式.
02.静态成员函数为整个类服务,而不是具体的一个类的实例服务.(这句话可能比较绕口,可以理解为在没有任何实例化的类对象的条件下调用类方法,详见上面代码注释处.)
03.静态成员函数中没有隐含的this指针,所以静态成员函数不可以操作类中的非静态成员.


ps:关于this指针的深入解释
在C++中,普通的成员函数一般都隐含了一个this指针,例如调用函数Fun(),实际上是this->Fun().
静态成员函数中没有这样的this指针,所以静态成员函数不能操作类中的非静态成员函数.否则编译器会报错.

三 注意事项
01.静态数据成员都是静态存储的,所以必须在main函数之前显示的对其进行初始化.
02.静态成员初始化与一般成员的初始化不同.
03.不能再头文件中声明静态全局变量,这点在简单的测试代码中无法体现,只有在多文件同时包含,引用和操作时候才会显露出来.其结果可能是产生了多个同名的静态数据.一旦出现这种问题,是非常难以查找和排除的.
04.不能将静态成员函数定义为虚函数.
05.静态成员函数没有this指针.
06.static缩短了子类对父类静态成员访问的时间,相对来说节省了内存空间
07.关于06条的补充,如果不想在子类中操作父类的静态成员,则可以在子类中定义一个同名的static成员.这样既可覆盖父类中的静态成员.并且根据C++的多态性变量命名规则.这样做是安全的.
08.静态成员声明在类中,操作在其外部,所以对其取地址操作就跟取普通成员的操作略有不同.静态变量地址是指向其数据类型的指针,函数地址则是一个类型为nonmember的函数指针.



### inline

[内联函数 (C++) | Microsoft Docs](https://docs.microsoft.com/zh-cn/cpp/cpp/inline-functions-cpp?view=msvc-170)

**`inline`** 关键字告诉编译器用函数定义中的代码替换每个函数调用实例。

使用内联函数可以使程序更快，因为它们消除了与函数调用关联的开销。 编译器可以通过对普通函数不可用的方式来优化内联扩展的函数。

内联代码替换在编译器的任意时间进行。 例如，如果采用该函数的地址或太大，则编译器不会内联函数。如果函数的地址太大，则不会内联。

类声明正文中定义的函数是隐式内联函数。

```bash
 备注
在类声明中，未声明 **`inline`** 关键字的函数。 可以在 **`inline`** 类声明中指定关键字;结果相同。
```

给定的内联成员函数在每个编译单元中必须以相同的方式进行声明。 此约束会导致内联函数像实例化函数一样运行。 此外，必须有精确的内联函数的定义。

类成员函数默认为外部链接，除非该函数的定义包含 **`inline`** 说明符。 前面的示例显示，无需使用 **`inline`** 说明符显式声明这些函数。 在 **`inline`** 函数定义中使用会导致它是内联函数。 但是，不允许在调用该函数后重新声明函数 **`inline`** 。

#### 何时使用内联函数

内联函数最适用于小函数使用，例如访问私有数据成员。 这两个单行或两行“访问器”函数的主要用途是返回有关对象的状态信息。 短函数对函数调用的开销很敏感。 较长的函数在调用和返回序列中花费的时间比例较低，并且从内联中受益更少。



[C++ inline 函数简介 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/375828786)

[C++ 中的 inline 用法 | 菜鸟教程 (runoob.com)](https://www.runoob.com/w3cnote/cpp-inline-usage.html)





### cast

c++强制类型转换运算符。

强制类型转换运算符是一种特殊的运算符，它把一种数据类型转换成另一种数据类型。强制转换运算符是一元运算符，它的优先级与其他一元运算符相同。

大多数的c++编译器都支持大部分通用的强制转换运算符。

```cpp
(type) expression
```

其中，type是转换后的数据类型。下面列出了c++支持的其他几种强制转换运算符：

- **const_cast<type> (expr)** : const_cast运算符用于修改类型的const/volatile属性。除了const或colatile属性外，目标类型必须与源类型相同。这种类型转换主要是用来操作所传对象的const属性，可以加上const属性，也可以去掉const属性。
- **dynamic_cast<type>(expr)** : dynamic_cast在运行时执行转换，验证转换的有效性。如果转换未执行，则转换失败，表达式expr倍判定为null。dynamic_cast执行动态转换是，type必须是类的指针、类的引用或者void*，如果type是类指针类型，那么expr也必须是一个指针，如果type是一个引用，那么expr也必须是一个引用。
- **reinterpret_cast<type>(expr)** : reinterpret_cast运算符把某种指针改为其他类型的指针。它可以把一个指针转换为一个整数，也可以把一个整数转换为一个指针。
- **static_cast<type>(expr)** : static_cast运算符执行非动态转换，没有运行时类检查来保证转换的安全性。例如，它可以把一个基类指针转换为派生类指针。



[(1条消息) C++ 四种cast 详解_帅B猪的博客-CSDN博客_c++ cast](https://blog.csdn.net/m0_46210273/article/details/121147406)



#### C++ 四种cast 详解



##### cast出现的意义

1.C++继承并扩展C语言的传统[类型转换](https://so.csdn.net/so/search?q=类型转换&spm=1001.2101.3001.7020)方式，提供了功能更加强大的转型机制（检查与风险）
2.更好的定位转型的地方。



##### reinterpret_cast

reinterpret_cast是四种强制转换中功能最为强大的（最暴力，最底层，最不安全）。它的本质是[编译器](https://so.csdn.net/so/search?q=编译器&spm=1001.2101.3001.7020)的指令。
它的作用：它可以把一个指针转换成一个整数，也可以把一个整数转换成一个指针。或者不同类型的指针的相互替换
代码示例：

```cpp
#include <iostream> 
int main()
{
	double a = 1.1;
	char * c = reinterpret_cast<char*>(&a);
	double* b = reinterpret_cast<double*>(c);
	printf("%lf",*b);
}

```

运行结果：

![image-20220904162545884](C++Note.assets/image-20220904162545884.png)



分析：我们通过中间的 char*来转double但是没有出现精度问题。事实上reinterpret_cast只是在编译器进行的予以转化，只要是个地址就可以转（二进制拷贝）。

##### const_cast

有两个功能，去掉const和加上const
加上const的情况：
代码：

```cpp
#include <iostream> 
int main()
{
	int* a=new int(1);
	const int * b = const_cast<const int * >(a);
	*a=2;
	//*b=2;，常量不能修改
	printf("%d\n",*a);
	printf("%d\n",*b);
	std::cout << a << std::endl;
	std::cout << b << std::endl;
}

```



![image-20220904205159453](C++Note.assets/image-20220904205159453.png)

分析：
我们发现值是一样的.

去掉const的情况（这里的情况非常多，深拷贝和浅拷贝各不一样，转型之后返回的可能不是原地址）：
代码：

```cpp
#include <iostream> 
class A
{
public:
	int num;
	A(int val = 100):num(val){}
	~A(){}
};
int main()
{
	//1.const 修饰指针对象,指向原对象 
	const A * pa1 = new A(200);
	A * cast_pa1 = const_cast<A * >(pa1);
	printf("1.const 修饰指针指向对象\n");
	printf("%p\n",pa1); 
	printf("%p\n",cast_pa1); 
	
	//2.const 修饰指向指针对象的值,指向原对象
	A * const pa2 = new A(200);
	A * cast_pa2 = const_cast<A * >(pa2);
	printf("2.const 修饰指向对象的值\n");
	printf("%p\n",pa2); 
	printf("%p\n",cast_pa2);
	
	//3.const 同时修饰指针对象和指针对象的值,指向原对象
	const A * const pa3 = new A(200);
	A * cast_pa3_1 = const_cast<A * >(pa3);
	const A * cast_pa3_2 = const_cast<A * >(pa3);
	A * const cast_pa3_3 = const_cast<A * >(pa3);
	printf("3.const 同时修饰指针对象和指针对象的值,指向原对象\n");
	printf("%p\n",pa3); 
	printf("%p\n",cast_pa3_1);
	printf("%p\n",cast_pa3_2);
	printf("%p\n",cast_pa3_3);
	
	//4.const 修饰普通对象，并且赋值给一般对象,不指向原对象 
	const A pa4;
	A cast_pa4 = const_cast<A &>(pa4);
	printf("4.const 修饰普通对象，并且赋值给一般对象\n");
	printf("%p\n",&pa4); 
	printf("%p\n",&cast_pa4);
	
	//5.const 修饰普通对象，并且赋值给引用对象,指向原对象
	const A pa5;
	A& cast_pa5 = const_cast<A& >(pa5);
	printf("5.const 修饰普通对象，并且赋值给引用对象\n");
	printf("%p\n",&pa5); 
	printf("%p\n",&cast_pa5);
	
	// 6. const 修饰对象，对象指针去 const 属性后赋给指针,指向原对象 
	const A pa6;
	A * cast_pa6 = const_cast<A * >(&pa6); 
	printf("6. const 修饰对象，对象指针去 const 属性后赋给指针\n");
	printf("%p\n",&pa6); 
	printf("%p\n",cast_pa6);
	
	//7.const修饰局部变量，不指向原对象
	const int pa7=1;
	int  cast_pa7_1 = const_cast<int&>(pa7); 
	int& cast_pa7_2 = const_cast<int&>(pa7);
	int* cast_pa7_3 = const_cast<int*>(&pa7);
	
	printf("6. const 修饰对象，对象指针去 const 属性后赋给指针\n");
	printf("%p\n",&pa7); 
	printf("%p\n",&cast_pa7_1);
	printf("%p\n",&cast_pa7_2);
	printf("%p\n",cast_pa7_3);
	cast_pa7_1=10;
	printf("%d,未修改\n",pa7);
	cast_pa7_2=100;
	printf("%d,未修改\n",pa7);
	*cast_pa7_3=1000;
	printf("%d,未修改\n",pa7);
}
```

执行结果：

![image-20220904205427023](C++Note.assets/image-20220904205427023.png)



分析：
去掉对象指针的const，全是原对象
去掉一般对象的const，如果赋值给一般对象则是新对象，否则全是原对象
去掉局部变量的const，全是新对象

##### static_cast

作用：
1.基本类型之间的转换
2.void指针转换为任意基本类型的指针
3.用于有继承关系的子类与父类之间的指针或引用的转换

基本类型之间的转换：
代码：

```cpp
#include <iostream> 
int main()
{
	double i=1.1;
	int a = static_cast<int>(i);
	printf("%d\n",a); 
	double b = static_cast<int>(a);
	printf("%lf\n",b);
}
```

运行结果：

![image-20220904210844814](C++Note.assets/image-20220904210844814.png)



分析：可以进行基本类型的转化，但是会损失精度类似与C语言的强制转化。跟reinterpret_cast不太一样reinterpret_cast是底层二进制的强制拷贝和语义转换不会损失精度。

void指针和其他指针的转换;
代码;

```cpp
#include <iostream> 
int main()
{
	int *a = new int(1);
	void *v = static_cast<void *>(a);
	int *p = static_cast<int *>(v);
	*a=2;
	printf("%d\n",*a);
	printf("%d\n",*p);
	printf("%p\n",a); 
	printf("%p\n",p); 
}
```

运行结果：

![image-20220904211005694](C++Note.assets/image-20220904211005694.png)



分析：这里是void指针和其他类型的指针进行的转化，结果是指向的是原地址。（普通类型的转换不是）

子类和父类之间的转换：
代码：

```cpp
#include <iostream> 
using namespace std;
class A
{
	public:
	A(){}
	void foo()
	{
		cout<<"A!"<<endl;
	}
};
class B:public A
{
	public:
	B(){}	
	void foo()
	{
		cout<<"B!"<<endl;
	}
}; 
int main()
{
	A *a = new A();
	B * b = static_cast<B *>(a);
	b->foo();
	return 0;
}
```

运行结果：

![image-20220904211228607](C++Note.assets/image-20220904211228607.png)



这是向下转型，是不安全的，但是为什么没有报错呢，因为B中还没有B特有的（B的成员变量）。我们在看看别的
代码：

```cpp
#include <iostream> 
using namespace std;
class A
{
	public:
	A(){}
	void foo()
	{
		cout<<"A!"<<endl;
	}
};
class B:public A
{
	char b='c';
	public:
	B(){}	
	void foo()
	{
		cout<<b<<endl;
	}
}; 
int main()
{
	A * a = new A();
	a->foo();
	static_cast<B*>(a)->foo();
	return 0;
}
```

运行结果;

![image-20220904211433590](C++Note.assets/image-20220904211433590.png)



分析：这里就发生了错误了，B中特有的成员变量没有初始化（使用了不安全的向下转型）
static_cast的类的转型类似于普通的强转，可以抛出异常

##### dynamic_cast

dynamic_cast用于类继承层次间的指针或引用转换(主要用于向下的安全转换)
dynamic_cast向下转型的安全性主要体现在RTTI
**RTTI**：
运行时类型识别。程序能够使用基类的指针或引用来检查着这些指针或引用所指的对象的实际派生类型（判断指针原型）
RTTI提供了两个非常有用的操作符：typeid和dynamic_cast。（三个最主要的东西，dynamic_cast,typeid,type_info）
typeid:typeid函数（为type_info类的友元函数，为什么要这样呢？目的是防止创建type_info对象）的主要作用就是让用户知道当前的变量是什么类型的,它可以返回一个type_info的引用，可以获取类的名称和编码typeid重载了type_info中的==和!=可以用于判断两个类型是否相等
1）typeid识别静态类型
当typeid中的操作数是如下情况之一时，typeid运算符指出操作数的静态类型，即编译时的类型。
（1）类型名
（2）一个**基本类型**的变量
（3）一个**具体**的对象(非指针对象)
（4）一个指向 不含有virtual函数的类 对象的指针的解引用
（5）一个指向 不含有virtual函数的类 对象的引用
静态类型在程序的运行过程中并不会改变，所以并不需要在程序运行时计算类型，在编译时就能根据操作数的静态类型，推导出其类型信息。例如如下的代码片断，typeid中的操作数均为静态类型：

代码：
```cpp
#include <iostream> 
#include <typeinfo> 
using namespace std;
class X  {
	public:
		X()
		{
			
		}
		void func()
		{
			
		}
}; 
class XX : public X  {
	public:
	XX()
	{
		
	}
	void func()
	{
			
	}
}; 
class Y  {
	public:
	Y()
	{
		
	}
	void func()
	{
			
	}
}; 
 
int main()
{
    int n = 0;
    XX xx;
    Y y;
    Y *py = &y;
 
    // int和XX都是类型名
    cout << typeid(int).name() << endl;
    cout << typeid(XX).name() << endl;
    // n为基本变量
    cout << typeid(n).name() << endl;
    // xx所属的类虽然存在virtual，但是xx为一个具体的对象
    cout << typeid(xx).name() << endl;
    // py为一个指针，属于基本类型
    cout << typeid(py).name() << endl;
    // py指向的Y的对象，但是类Y不存在virtual函数
    cout << typeid(*py).name() << endl;
    return 0;
}
```



运行输出：

![image-20220904212436846](C++Note.assets/image-20220904212436846.png)



2）typeid识别多态类型
当typeid中的操作数是如下情况之一时，typeid运算符需要在程序运行时计算类型，因为其操作数的类型在编译时期是不能被确定的。
（1）一个指向含有virtual函数的类对象的指针的解引用
（2）一个指向含有virtual函数的类对象的引用

代码：

```cpp
#include <iostream> 
#include <typeinfo> 
using namespace std;
class X
{
    public:
        X()
        {
            mX = 101;
        }
        virtual void vfunc()
        {
            cout << "X::vfunc()" << endl;
        }
    private:
        int mX;
};
class XX : public X
{
    public:
        XX():
            X()
        {
            mXX = 1001;
        }
        virtual void vfunc()
        {
            cout << "XX::vfunc()" << endl;
        }
    private:
        int mXX;
};
void printTypeInfo(const X *px)
{
    cout << "typeid(px) -> " << typeid(px).name() << endl;
    cout << "typeid(*px) -> " << typeid(*px).name() << endl;
}
int main()
{
    X x;
    XX xx;
    printTypeInfo(&x);
    printTypeInfo(&xx);
    return 0;
}
```



运行结果：

![image-20220904213346528](C++Note.assets/image-20220904213346528.png)



最后真实的判断出了指针原型
那么问题来了，typeid是如何计算这个类型信息的呢？下面将重点说明这个问题。

多态类型是通过在类中声明一个或多个virtual函数来区分的。因为在C++中，一个具备多态性质的类，正是内含直接声明或继承而来的virtual函数。多态类的对象的类型信息保存在虚函数表的索引的-1的项中，该项是一个type_info对象的地址，该type_info对象保存着该对象对应的类型信息，每个类（多态）都对应着一个type_info对象
在多重继承和虚拟继承的情况下，一个类有n（n>1）个虚函数表，该类的对象也有n个vptr，分别指向这些虚函数表，但是一个类的所有的虚函数表的索引为-1的项的值（type_info对象的地址）都是相等的，即它们都指向同一个type_info对象，这样就实现了无论使用了哪一个基类的指针或引用指向其派生类的对象，都能通过相应的虚函数表获取到相同的type_info对象，从而得到相同的类型信息。

##### dynamic_cast（可以抛出异常）

dynamic_cast借助RTTI机制实现了安全的向下转型（无法转型的返回NULL）
代码：

```cpp
#include <iostream> 
#include <typeinfo> 
using namespace std;
class X
{
    public:
        X()
        {
            mX = 101;
        }
        virtual ~X()
        {
        }
    private:
        int mX;
};
 
class XX : public X
{
    public:
        XX():
            X()
        {
            mXX = 1001;
        }
        virtual ~XX()
        {
        }
    private:
        int mXX;
};
 
class YX : public X
{
    public:
        YX()
        {
            mYX = 1002;
        }
        virtual ~YX()
        {
        }
    private:
        int mYX;
};
int main()
{
    X x;
    XX xx;
    YX yx;
 
    X *px = &xx;
    cout << px << endl;
 
    XX *pxx = dynamic_cast<XX*>(px); // 转换1
    cout << pxx << endl;
 
    YX *pyx = dynamic_cast<YX*>(px); // 转换2
    cout << pyx << endl;
 
    pyx = (YX*)px; // 转换3
    cout << pyx << endl;
 
    pyx = static_cast<YX*>(px); // 转换4
    cout << pyx << endl;
 
    return 0;
}

```

运行结果：

![image-20220904213629226](C++Note.assets/image-20220904213629226.png)



分析：
px是一个基类（X）的指针，但是它指向了派生类XX的一个对象。在转换1中，转换成功，因为px指向的对象确实为XX的对象。在转换2中，转换失败，因为px指向的对象并不是一个YX对象，此时dymanic_cast返回NULL。转换3为C风格的类型转换而转换4使用的是C++中的静态类型转换，它们均能成功转换，但是这个对象实际上并不是一个YX的对象，所以在转换3和转换4中，若继续通过指针使用该对象必然会导致错误，所以这个转换是不安全的。

声明：引用的情况与指针稍有不同，失败时并不是返回NULL，而是抛出一个bad_cast异常，因为引用不能参考NULL。

### extern

[C++extern详解 - 骚猪mark - 博客园 (cnblogs.com)](https://www.cnblogs.com/markzhuqian/p/14461696.html)
[extern (C++) | Microsoft Docs](https://docs.microsoft.com/zh-cn/cpp/cpp/extern-cpp?view=msvc-170)



#### extern——关键字

extern是C语言中的一个关键字，一般用在变量名前或函数名前，作用是用来说明“**此变量/函数是在别处定义的，要在此处引用**”，extern这个关键字大部分读者应该是在变量的存储类型这一类的内容中

遇到的，下面先分析C语言不同的存储类型



在C语言中变量和函数有**数据类型**和**存储类型**两个属性，因此变量定义的一般形式为：存储类型 数据类型 变量名表；

C语言提供了一下几种不同的存储类型：

（1） 自动变量（auto）

（2） 静态变量（static）

（3） 外部变量（extern）

（4） 寄存器变量（register）

（上面的auto、static、extern、register都是C语言的关键字），这里只分析extern关键字的使用

外部变量（全局变量）extern----全局静态存储区

标准定义格式：**extern 类型名 变量名；**



1、函数的声明extern关键词是可有可无的，因为函数本身不加修饰的话就是extern。但是引用的时候一样需要声明的。

2、全局变量在外部使用声明时，extern关键字是必须的，如果变量没有extern修饰且没有显式的初始化，同样成为变量的定义，因此此时必须加extern，而编译器在此标记存储空间在执行时加载内并初始化为0。而局部变量的声明不能有extern的修饰，且局部变量在运行时才在堆栈部分分配内存。

3、全局变量或函数本质上讲没有区别，函数名是指向函数二进制块开头处的指针。而全局变量是在函数外部声明的变量。函数名也在函数外，因此函数也是全局的。

4、谨记：声明可以多次，定义只能一次。

5、extern int i; //声明，不是定义
int i; //声明，也是定义

##### 示例



```cpp
#include <stdio.h>

extern int count;

void write_extern(void)
{
	printf("count is %d\n", count);
}
```





```cpp
#include <stdio.h>

int count ;
extern void write_extern();

int main()
{
	count = 5;
    write_extern();
}
```



#### extern"C" 作用

[C/C++中extern关键字详解 - 简书 (jianshu.com)](https://www.jianshu.com/p/111dcd1c0201)

C++语言在编译的时候为了解决函数的多态问题，会将函数名和参数联合起来生成一个中间的函数名称，而C语言则不会，因此会造成链接时无法找到对应函数的情况，此时C函数就需要用extern “C”进行链接指定，这告诉编译器，请保持我的名称，不要给我生成用于链接的中间函数名。

比如说你用C 开发了一个DLL 库，为了能够让C ++语言也能够调用你的DLL 输出(Export) 的函数，你需要用extern "C" 来强制编译器不要修改你的函数名。

通常，在C 语言的头文件中经常可以看到类似下面这种形式的代码：

```cpp
#ifdef __cplusplus  
extern "C" {  
#endif  
  
/**** some declaration or so *****/  
  
#ifdef __cplusplus  
}  
#endif
```



1. 现在要写一个c语言的模块，供以后使用（以后的项目可能是c的也可能是c++的），源文件事先编译好，编译成.so或.o都无所谓。头文件中声明函数时要用条件编译包含起来，如下：

```cpp
#ifdef __cpluscplus  
extern "C" {  
#endif  
  
//some code  
  
#ifdef __cplusplus  
}  
#endif  
```

也就是把所有函数声明放在some code的位置。

2. 如果这个模块已经存在了，可能是公司里的前辈写的，反正就是已经存在了，模块的.h文件中没有extern "C"关键字，这个模块又不希望被改动的情况下，可以这样，在你的c++文件中，包含该模块的头文件时加上extern "C", 如下：



```cpp
extern "C" {  
#include "test_extern_c.h"  
} 
```

3. 上面例子中，如果仅仅使用模块中的1个函数，而不需要include整个模块时，可以不include头文件，而单独声明该函数，像这样:

```cpp

extern "C"{  
int ThisIsTest(int, int);            
} 
```



**注意:** 当单独声明函数时候， 就不能要头文件，或者在头文件中不能写extern intThisIsTest(int a, int b);否则会有error C2732: 链接规范与“ThisIsTest”的早期规范冲突，这个错误。

####  声明和定义知识点

1. 定义也是声明，extern声明不是定义，即不分配存储空间。extern告诉编译器变量在其他地方定义了。
    eg：extern int  i; //声明，不是定义
    int i; //声明，也是定义
2. 如果声明有初始化式，就被当作定义，即使前面加了extern。只有当extern声明位于函数外部时，才可以被初始化。
    eg：extern double pi=3.1416; //定义
3. 函数的声明和定义区别比较简单，带有{}的就是定义，否则就是声明。
    eg：extern double max(double d1,double d2); //声明
    double max(double d1,double d2){} //定义
4. 除非有extern关键字，否则都是变量的定义。
    eg：extern inti; //声明
    inti; //定义

注:  basic_stdy.h中有char
 glob_str[];而basic_stdy.cpp有char
 glob_str;此时头文件中就不是定义，默认为extern
 **程序设计风格：**

1. 不要把变量定义放入.h文件，这样容易导致重复定义错误。
2. 尽量使用static关键字把变量定义限制于该源文件作用域，除非变量被设计成全局的。
    也就是说
3. 可以在头文件中声明一个变量，在用的时候包含这个头文件就声明了这个变量。



### 堆和栈

[(2条消息) C++中堆（heap）和栈(stack)的区别（面试中被问到的题目）_Howie_Yue的博客-CSDN博客](https://blog.csdn.net/qq_34175893/article/details/83502412)



一般面试官想问的是C++的内存分区管理方式。

首先说明,在C++中，内存分为5个区：堆、占、自由存储区、全局/静态存储区、常量存储区

- **栈**：是由编译器在需要时自动分配，不需要时自动清除的变量存储区。通常存放局部变量、函数参数等。
- **堆**：是由new分配的内存块，由程序员释放（编译器不管），一般一个new与一个delete对应，一个new[]与一个delete[]对应。如果程序员没有释放掉，        资源将由操作系统在程序结束后自动回收。
- **自由存储区**：是由malloc等分配的内存块，和堆十分相似，用free来释放。
- **全局/静态存储区**：全局变量和静态变量被分配到同一块内存中（在C语言中，全局变量又分为初始化的和未初始化的，C++中没有这一区分）。
- **常量存储区**：这是一块特殊存储区，里边存放常量，不允许修改。

#### 堆和栈的区别



![image-20220830224711171](C++Note.assets/image-20220830224711171.png)



## union介绍

　　共用体，也叫联合体，在一个“联合”内可以定义多种不同的数据类型， 一个被说明为该“联合”类型的变量中，允许装入该“联合”所定义的任何一种数据，这些数据共享同一段内存，以达到节省空间的目的。**union变量所占用的内存长度等于最长的成员的内存长度。**

 **union与struct比较**

先看一个关于struct的例子：

```cpp
struct student
{
     char mark;
     long num;
     float score;
};
```

其struct的内存结构如下，sizeof(struct student)的值为12bytes。

下面是关于union的例子：

```cpp
union test
{
     char mark;
     long num;
     float score;
};
```

sizeof(union test)的值为4。因为共用体将一个char类型的mark、一个long类型的num变量和一个float类型的score变量存放在**同一个地址开始的内存单元**中，而char类型和long类型所占的内存字节数是不一样的，但是在union中都是从同一个地址存放的，也就是使用的覆盖技术，这三个变量互相覆盖，而这种使几个不同的变量共占同一段内存的结构，称为“共用体”类型的结构。其union类型的结构如下：

**因union中的所有成员起始地址都是一样的，所以&a.mark、&a.num和&a.score的值都是一样的。**

 不能如下使用：

```cpp
union test a;
printf("%d", a); //错误
```

由于a的存储区有好几种类型，分别占不同长度的存储区，仅写共用体变量名a，这样使编译器无法确定究竟输出的哪一个成员的值。

```cpp
printf("%d", a.mark);  //正确
```



**测试大小端**

union的一个用法就是可以用来测试CPU是大端模式还是小端模式：

```cpp
#include <iostream>
using namespace std;

void checkCPU()
{
    union MyUnion{
        int a;
        char c;
    }test;
    test.a = 1;
    if (test.c == 1)
        cout << "little endian" <<endl;
    else cout << "big endian" <<endl;
}

int main()
{
    checkCPU();
    return 0;
}
```

举例，代码如下：

```cpp
#include <iostream>
using namespace std;

union test
{
     char mark;
     long num;
     float score;
}a;

int main()
{
     // cout<<a<<endl; // wrong
     a.mark = 'b';
     cout<<a.mark<<endl; // 输出'b'
     cout<<a.num<<endl; // 98 字符'b'的ACSII值
     cout<<a.score<<endl; // 输出错误值

     a.num = 10;
     cout<<a.mark<<endl; // 输出换行 非常感谢suxin同学的指正
     cout<<a.num<<endl; // 输出10
     cout<<a.score<<endl; // 输出错误值

     a.score = 10.0;
     cout<<a.mark<<endl; // 输出空
     cout<<a.num<<endl; // 输出错误值
     cout<<a.score<<endl; // 输出10

     return 0;
}
```

**C++中union**

上面总结的union使用法则，在C++中依然适用。如果加入对象呢？

```cpp
#include <iostream>
using namespace std;

class CA
{
     int m_a;
};

union Test
{
     CA a;
     double d;
};

int main()
{
     return 0;
}
```

上面代码运行没有问题。

　　如果在类CA中**添加了构造函数，或者添加析构函数，就会发现程序会出现错误。**由于union里面的东西共享内存，所以不能定义静态、引用类型的变量。由于在union里也不允许存放带有构造函数、析构函数和复制构造函数等的类的对象，但是可以存放对应的类对象指针。编译器无法保证类的构造函数和析构函数得到正确的调用，由此，就可能出现内存泄漏。所以，在C++中使用union时，尽量保持C语言中使用union的风格，尽量不要让union带有对象。



## 线程

### CreateThread 创建线程



线程是进程中的一个实体，是被系统独立调度和分派的基本单位。一个进程可以拥有多个线程，但是一个线程必须有一个进程。线程自己不拥有系统资源，只有运行所必须的一些[数据结构](https://so.csdn.net/so/search?q=数据结构&spm=1001.2101.3001.7020)，但它可以与同属于一个进程的其它线程共享进程所拥有的全部资源，同一个进程中的多个线程可以并发执行。

在C/C++中可以通过CreateThread函数在进程中创建线程，函数的具体格式如下：

D:\Windows Kits\10\Include\10.0.19041.0\um\processthreadsapi.h

```cpp
WINBASEAPI
_Ret_maybenull_
HANDLE
WINAPI
CreateThread(
    _In_opt_ LPSECURITY_ATTRIBUTES lpThreadAttributes,
    _In_ SIZE_T dwStackSize,
    _In_ LPTHREAD_START_ROUTINE lpStartAddress,
    _In_opt_ __drv_aliasesMem LPVOID lpParameter,
    _In_ DWORD dwCreationFlags,
    _Out_opt_ LPDWORD lpThreadId
    );
```

参数的含义如下：
```bash
lpThreadAttrivutes：指向SECURITY_ATTRIBUTES的指针，用于定义新线程的安全属性，一般设置成NULL；

dwStackSize：分配以字节数表示的线程堆栈的大小，默认值是0；

lpStartAddress：指向一个线程函数地址。每个线程都有自己的线程函数，线程函数是线程具体的执行代码；

lpParameter：传递给线程函数的参数；

dwCreationFlags：表示创建线程的运行状态，其中CREATE_SUSPEND表示挂起当前创建的线程，而0表示立即执行当前创建的进程；

lpThreadID：返回新创建的线程的ID编号；
```

- 第一个参数 `lpThreadAttributes` 表示线程内核对象的安全属性，一般传入NULL表示使用默认设置。
- 第二个参数 `dwStackSize` 表示线程栈空间大小。传入0表示使用默认大小（1MB）。
- 第三个参数 `lpStartAddress` 表示新线程所执行的线程函数地址，多个线程可以使用同一个函数地址。
- 第四个参数 `lpParameter` 是传给线程函数的参数。
- 第五个参数 `dwCreationFlags` 指定额外的标志来控制线程的创建，为0表示线程创建之后立即就可以进行调度，如果为CREATE_SUSPENDED则表示线程创建后暂停运行，这样它就无法调度，直到调用ResumeThread()。
- 第六个参数 `lpThreadId` 将返回线程的ID号，传入NULL表示不需要返回该线程ID号。



如果函数调用成功，则返回新线程的句柄，调用WaitForSingleObject函数等待所创建线程的运行结束。函数的格式如下：

```cpp
DWORD WaitForSingleObject(
                          HANDLE hHandle,
                          DWORD dwMilliseconds
                         );
```

参数的含义如下：

```bash
hHandle：指定对象或时间的句柄；
dwMilliseconds：等待时间，以毫秒为单位，当超过等待时间时，此函数返回。如果参数设置为0，则该函数立即返回；如果设置成INFINITE，则该函数直到有信号才返回。
```

一般情况下需要创建多个线程来提高程序的执行效率，但是多个线程同时运行的时候可能调用线程函数，在多个线程同时对一个内存地址进行写入操作，由于CPU时间调度的问题，写入的数据会被多次覆盖，所以要使线程同步。

就是说，当有一个线程对文件进行操作时，其它线程只能等待。可以通过临界区对象实现线程同步。临界区对象是定义在数据段中的一个CRITICAL_SECTION结构，Windows内部使用这个结构记录一些信息，确保同一时间只有一个线程访问改数据段中的数据。

使用临界区的步骤如下：

（1）初始化一个CRITICAL_SECTION结构；在使用临界区对象之前，需要定义全局CRITICAL_SECTION变量，在调用CreateThread函数前调用InitializeCriticalSection函数初始化临界区对象；

（2）申请进入一个临界区；在线程函数中要对保护的数据进行操作前，可以通过调用EnterCriticalSection函数申请进入临界区。由于同一时间内只能有一个线程进入临界区，所以在申请的时候如果有一个线程已经进入临界区，则该函数就会一直等到那个线程执行完临界区代码；

（3）离开临界区；当执行完临界区代码后，需要调用LeaveCriticalSection函数离开临界区；

（4）删除临界区；当不需要临界区时调用DeleteCriticalSection函数将临界区对象删除；

下面的代码创建了5个线程，每个线程在文件中写入10000个“hello”：

```cpp
#include <stdio.h>
#include <windows.h>
#include <tchar.h>
HANDLE hFile;
CRITICAL_SECTION cs;//定义临界区全局变量
//线程函数：在文件中写入10000个hello
DWORD WINAPI Thread(LPVOID lpParam)
{
	long long n = (long long)lpParam;
	DWORD dwWrite;
	for (int i = 0;i < 10000;i++)
	{
		//进入临界区
		EnterCriticalSection(&cs);
		char data[512] = "hello\r\n";
		//写文件
		WriteFile(hFile, data, strlen(data), &dwWrite, NULL);
		//离开临界区
		LeaveCriticalSection(&cs);
	}
	printf("Thread #%d returned successfully\n", n);
	return 0;
}
int main()
{
	char *filename = "hack.txt";
	WCHAR name[20] = { 0 };
	MultiByteToWideChar(CP_ACP, 0, filename, strlen(filename) + 1, name, sizeof(name) / sizeof(name[0]));
	//创建文件
	hFile = CreateFile(filename, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
	if (hFile == INVALID_HANDLE_VALUE)
	{
		printf("CreateFile error.\n");
		return 0;
	}
	DWORD ThreadID;
	HANDLE hThread[5];
	//初始化临界区
	InitializeCriticalSection(&cs);
	for (int i = 0;i < 5;i++)
	{
		//创建线程，并调用Thread写文件
		hThread[i] = CreateThread(NULL, 0, Thread, (LPVOID)(i + 1), 0, &ThreadID);
		printf("Thread #%d has been created successfully.\n", i + 1);
	}
	//等待所有进程结束
	WaitForMultipleObjects(5, hThread, TRUE, INFINITE);
	//删除临界区
	DeleteCriticalSection(&cs);
	//关闭文件句柄
	CloseHandle(hFile);
	return 0;
}
```

![image-20221124153123681](C++Note.assets/image-20221124153123681.png)



**测试程序2**

```cpp


/* 创建第一个线程。主进程结束，则撤销线程。 */

#include<Windows.h>
#include<stdio.h>

DWORD WINAPI ThreadFunc(LPVOID);

int main()
{
	HANDLE hThread;
	DWORD  threadId;

	hThread = CreateThread(NULL, 0,	ThreadFunc, 0, 0, &threadId); // 创建线程
	printf("我是主线程， pid = %d\n", GetCurrentThreadId());  //输出主线程pid
	Sleep(2000);
}

DWORD WINAPI ThreadFunc(LPVOID p)
{	
	printf("我是子线程， pid = %d\n", GetCurrentThreadId());   //输出子线程pid
	return 0;
}
```

运行输出：

```bash
我是主线程， pid = 7344
我是子线程， pid = 21108

--------------------------------
Process exited after 2.053 seconds with return value 0

Press ANY key to exit...
```





### 更安全的创建线程方式_beginthreadex()

[更安全的创建线程方式_beginthreadex()](https://www.cnblogs.com/ay-a/p/9135652.html)

CreateThread()函数是Windows提供的API接口，在C/C++语言另有一个创建线程的函数_beginthreadex()，我们应该尽量使用_beginthreadex()来代替使用CreateThread()，因为它比CreateThread()更安全。

其原因首先要从标准C运行库与多线程的矛盾说起，标准C运行库在1970年被实现了，由于当时没任何一个操作系统提供对多线程的支持。因此编写标准C运行库的程序员根本没考虑多线程程序使用标准C运行库的情况。比如标准C运行库的全局变量errno。很多运行库中的函数在出错时会将错误代号赋值给这个全局变量，这样可以方便调试。

但如果有这样的一个代码片段：

```cpp
if (system("notepad.exe readme.txt") == -1)  
{  
    switch(errno)  
    {  
        ...//错误处理代码  
    }  
}  
```

假设某个线程A在执行上面的代码，该线程在调用system()之后且尚未调用switch()语句时另外一个线程B启动了，这个线程B也调用了标准C运行库的函数，不幸的是这个函数执行出错了并将错误代号写入全局变量errno中。这样线程A一旦开始执行switch()语句时，它将访问一个被B线程改动了的errno。这种情况必须要加以避免！因为不单单是这一个变量会出问题，其它像strerror()、strtok()、tmpnam()、gmtime()、asctime()等函数也会遇到这种由多个线程访问修改导致的数据覆盖问题。

为了解决这个问题，Windows操作系统提供了这样的一种解决方案——每个线程都将拥有自己专用的一块内存区域来供标准C运行库中所有有需要的函数使用。而且这块内存区域的创建就是由C/C++运行库函数_beginthreadex()来负责的。

_beginthreadex()函数在创建新线程时会分配并初始化一个_tiddata块。这个_tiddata块自然是用来存放一些需要线程独享的数据。新线程运行时会首先将_tiddata块与自己进一步关联起来。然后新线程调用标准C运行库函数如strtok()时就会先取得_tiddata块的地址再将需要保护的数据存入_tiddata块中。这样每个线程就只会访问和修改自己的数据而不会去篡改其它线程的数据了。因此，如果在代码中有使用标准C运行库中的函数时，尽量使用`_beginthreadex()`来代替CreateThread()。

**实例**

下面的例子使用_beginthreadex()来创建线程。

```cpp
#include<process.h>
#include<windows.h>
#include<iostream>
using namespace std;

unsigned int __stdcall ThreadFun(PVOID pM)
{
	printf("线程ID 为 %d 的子线程输出： Hello World\n", GetCurrentThreadId());
	return 0;
}


int main()
{
	const int THREAD_NUM = 5;
	HANDLE handle[THREAD_NUM];
	for (int i = 0; i < THREAD_NUM; i++)
		handle[i] = (HANDLE)_beginthreadex(NULL, 0, ThreadFun, NULL, 0, NULL);
	WaitForMultipleObjects(THREAD_NUM, handle, TRUE, INFINITE);
	return 0;
}

```

运行输出：

```bash
线程ID 为 27396 的子线程输出： Hello World
线程ID 为 25560 的子线程输出： Hello World
线程ID 为 27432 的子线程输出： Hello World
线程ID 为 7504 的子线程输出： Hello World
线程ID 为 26896 的子线程输出： Hello World

--------------------------------
Process exited after 0.1729 seconds with return value 0

Press ANY key to exit...
```



### Spinlock 自旋锁

自旋锁与互斥锁有点类似，只是自旋锁不会引起调用者睡眠，如果自旋锁已经被别的执行单元保持，调用者就一直循环在那里看是否该自旋锁的保持者已经释放了锁，"自旋"一词就是因此而得名。

　　由于自旋锁使用者一般保持锁时间非常短，因此选择自旋而不是睡眠是非常必要的，自旋锁的效率远高于互斥锁。

[spinlock前世今生 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/133445693)





## 进程





## 并行



## 并发



## atomic原子操作

### atomic概述

所谓[原子操作](https://so.csdn.net/so/search?q=原子操作&spm=1001.2101.3001.7020)，就是多线程程序中“最小的且不可并行化的”操作。对于在多个线程间共享的一个资源而言，这意味着同一时刻，多个线程中有且仅有一个线程在对这个资源进行操作，即互斥访问。

C++ 11 新增atomic可以实现原子操作

### 非原子操作



```cpp
#include <thread>
#include <atomic>
#include <iostream>
using namespace std;

int i = 0;
const int maxCnt = 1000000;
void mythread()
{
    for (int j = 0; j < maxCnt; j++)
        i++;  //线程同时操作变量
}

int main()
{
    auto begin = chrono::high_resolution_clock::now();
    thread t1(mythread);
    thread t2(mythread);
    t1.join();
    t2.join();
    auto end = chrono::high_resolution_clock::now();
    cout << "i=" << i << endl;
    cout << "time: " << chrono::duration_cast<chrono::microseconds>(end - begin).count() * 1e-6 << "s" << endl; //秒计时
}
```

运行输出：

```bash
i=1020827
time: 0.007223s

--------------------------------
Process exited after 0.05536 seconds with return value 0

Press ANY key to exit...
```

测试结果：发现结果并不是2000000，两个线程同时对共享资源操作会出问题

问题分析：以下是i++反汇编代码

![image-20221125000032345](C++Note.assets/image-20221125000032345.png)



i++这一条程序在计算机中是分几个机器指令来执行的，先把i值赋值给eax寄存器，eax寄存器自加1，然后再把eax寄存器值赋值回i，如果在指令执行过程中发生了线程调度，那么这一套完整的i++指令操作被打断，会发生结果错乱。

举例子：

![image-20221125000247759](C++Note.assets/image-20221125000247759.png)



从图上可知，EAX寄存器进行了两次自加操作，但实际上i的值只加了1

### 加锁



```cpp
#include <thread>
#include <atomic>
#include <iostream>
#include<mutex>
using namespace std;

int i = 0;
const int maxCnt = 1000000;
mutex mut;
void mythread()
{
    for (int j = 0; j < maxCnt; j++)
    {
        mut.lock();    //加锁操作
        i++;
        mut.unlock();
    }
}

int main()
{

    auto begin = chrono::high_resolution_clock::now();
    thread t1(mythread);
    thread t2(mythread);
    t1.join();
    t2.join();
    auto end = chrono::high_resolution_clock::now();
    cout << "i=" << i << endl;
    cout << "time: " << chrono::duration_cast<chrono::microseconds>(end - begin).count() * 1e-6 << "s" << endl; //秒计时
}
```

运行结果：

```bash
i=2000000
time: 0.047453s

--------------------------------
Process exited after 0.07993 seconds with return value 0

Press ANY key to exit...
```

测试结果如图：虽然保证了结果正确，但耗时也增加了

### atomic代码

使用方法：

atomic<int> i;
1
对应i++汇编代码如下：
发现调用的atomic类方法operator++，继续追踪。
只需关注红笔标注的代码，lock xadd 指令就是计算机硬件底层提供的原子性支持。

代码实现：

```cpp
#include <thread>
#include <atomic>
#include <iostream>
#include<mutex>
using namespace std;

atomic<int> i;
//atomic_int32_t i;  两种写法
const int maxCnt = 1000000;
void mythread()
{
    for (int j = 0; j < maxCnt; j++)
    {
        i++;
    }
}

int main()
{

    auto begin = chrono::high_resolution_clock::now();
    thread t1(mythread);
    thread t2(mythread);
    t1.join();
    t2.join();
    auto end = chrono::high_resolution_clock::now();
    cout << "i=" << i << endl;
    cout << "time: " << chrono::duration_cast<chrono::microseconds>(end - begin).count() * 1e-6 << "s" << endl; //秒计时
}
```

运行结果：

测试结果如下：atomic保证原子性操作的同时，耗时也较低

```bash
i=2000000
time: 0.020719s

--------------------------------
Process exited after 0.05644 seconds with return value 0

Press ANY key to exit...
```



## OpenMP与TBB

### OpenMP简介

OpenMP是一种用于共享内存并行系统的多线程程序设计方案，支持的编程语言包括C、C++和Fortran。OpenMP提供了对并行算法的高层抽象描述，特别适合在多核CPU机器上的并行程序设计。编译器根据程序中添加的pragma指令，自动将程序并行处理，使用OpenMP降低了并行编程的难度和复杂度。当编译器不支持OpenMP时，程序会退化成普通（串行）程序。程序中已有的OpenMP指令不会影响程序的正常编译运行。



### TBB简介

TBB(Thread Building Blocks)是英特尔发布的一个库，全称为 Threading Building Blocks。TBB 获得过 17 届 Jolt Productivity Awards，是一套 C++ 模板库，和直接利用 OS API 写程序的 raw thread 比，在并行编程方面提供了适当的抽象，当然还包括更多其他内容，比如 task 概念，常用算法的成熟实现，自动负载均衡特 性还有不绑定 CPU 数量的灵活的可扩展性等等。在多核的平台上开发并行化的程序，必须合理地利用系统的资源 - 如与内核数目相匹配的线程，内存的合理访问次序，最大化重用缓存。有时候用户使用(系统)低级的应用接口创建、管理线程，很难保证是否程序处于最佳状态。 而 TBB 很好地解决了上述问题： 

 1）TBB提供C++模版库，用户不必关注线程，而专注任务本身。 
 2）抽象层仅需很少的接口代码，性能上毫不逊色。 
 3）灵活地适合不同的多核平台。 
 4）线程库的接口适合于跨平台的移植(Linux, Windows, Mac) 

OneTBB源码： https://github.com/oneapi-src/oneTBB

OneTBB开发手册： https://oneapi-src.github.io/oneTBB/





