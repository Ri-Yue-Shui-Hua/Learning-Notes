# C++模板

[C++ 模板 | 菜鸟教程 (runoob.com)](https://www.runoob.com/cplusplus/cpp-templates.html)



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



#  C++ STL

参考：[【C/C++】STL详解](https://blog.csdn.net/qq_42322103/article/details/99685797)



## 概述

长久以来，软件界一直希望建立一种可重复利用的东西，以及一种得以制造出”可重复运用的东西”的方法，从函数(functions)，类别(classes),函数库(function libraries),类别库(class libraries)、各种组件，从模块化设计，到面向对象(object oriented )，为的就是复用性的提升。

复用性必须建立在某种标准之上。但是在许多环境下，就连软件开发最基本的数据结构(data structures) 和算法(algorithm)都未能有一套标准。大量程序员被迫从事大量重复的工作，竟然是为了完成前人已经完成而自己手上并未拥有的程序代码，这不仅是人力资源的浪费，也是挫折与痛苦的来源。

为了建立数据结构和算法的一套标准，并且降低他们之间的耦合关系，以提升各自的独立性、弹性、交互操作性(相互合作性,interoperability),诞生了STL。

STL(Standard Template Library,标准模板库)，是惠普实验室开发的一系列软件的统称。现在主要出现在 c++中，但是在引入 c++之前该技术已经存在很长时间了。

STL 从广义上分为: **容器**(container) **算法**(algorithm) **迭代器**(iterator)。

容器和算法之间通过迭代器进行无缝连接。STL 几乎所有的代码都采用了模板类或者模板函数，这相比传统的由函数和类组成的库来说提供了更好的代码重用机会。

STL(Standard Template Library)标准模板库,在我们 c++标准程序库中隶属于 STL 的占到了 80%以上。


## STL六大组件简介

STL提供了六大组件，彼此之间可以组合套用，这六大组件分别是:**容器、算法、迭代器、仿函数、适配器（配接器）、空间配置器**。

**容器**：各种数据结构，如vector、list、deque、set、map等,用来存放数据，从实现角度来看，STL容器是一种class template。

**算法**：各种常用的算法，如sort、find、copy、for_each。从实现的角度来看，STL算法是一种function tempalte.

**迭代器**：扮演了容器与算法之间的胶合剂，共有五种类型，从实现角度来看，迭代器是一种将operator* , operator-> , operator++,operator–等指针相关操作予以重载的class template. 所有STL容器都附带有自己专属的迭代器，只有容器的设计者才知道如何遍历自己的元素。原生指针(native pointer)也是一种迭代器。

**仿函数**：行为类似函数，可作为算法的某种策略。从实现角度来看，仿函数是一种重载了operator()的class 或者class template

**适配器**：一种用来修饰容器或者仿函数或迭代器接口的东西。

**空间配置器**：负责空间的配置与管理。从实现角度看，配置器是一个实现了动态空间配置、空间管理、空间释放的class tempalte.

STL六大组件的交互关系，容器通过空间配置器取得数据存储空间，算法通过迭代器存储容器中的内容，仿函数可以协助算法完成不同的策略的变化，适配器可以修饰仿函数。


 **STL的优点**

- STL 是 C++的一部分，因此不用额外安装什么，它被内建在你的编译器之内。
- STL 的一个重要特性是将数据和操作分离。数据由容器类别加以管理，操作则由可定制的算法定义。迭代器在两者之间充当“粘合剂”,以使算法可以和容器交互运作。
- 程序员可以不用思考 STL 具体的实现过程，只要能够熟练使用 STL 就 OK 了。这样他们就可以把精力放在程序开发的别的方面。
- STL 具有高可重用性，高性能，高移植性，跨平台的优点。
  高可重用性：STL 中几乎所有的代码都采用了模板类和模版函数的方式实现，这相比于传统的由函数和类组成的库来说提供了更好的代码重用机会。
  高性能：如 map 可以高效地从十万条记录里面查找出指定的记录，因为 map 是采用红黑树的变体实现的。
  高移植性：如在项目 A 上用 STL 编写的模块，可以直接移植到项目 B 上。

## 三大组件介绍



### 容器

几乎可以说，任何特定的数据结构都是为了实现某种特定的算法。STL容器就是将运用最广泛的一些数据结构实现出来。
常用的数据结构：**数组**(array) , **链表**(list), **树**(tree)，**栈**(stack), **队列**(queue), **集合**(set),**映射表**(map), 根据数据在容器中的排列特性，这些数据分为**序列式容器**和**关联式容器**两种。

- 序列式容器强调值的排序，序列式容器中的每个元素均有固定的位置，除非用删除或插入的操作改变这个位置。Vector容器、Deque容器、List容器等。
- 关联式容器是非线性的树结构，更准确的说是二叉树结构。各元素之间没有严格的物理上的顺序关系，也就是说元素在容器中并没有保存元素置入容器时的逻辑顺序。关联式容器另一个显著特点是：在值中选择一个值作为关键字key，这个关键字对值起到索引的作用，方便查找。Set/multiset容器 Map/multimap容器.

### 算法

算法，问题的解法，以有限的步骤，解决逻辑或数学上的问题。

我们所编写的每个程序都是一个算法，其中的每个函数也都是一个算法，毕竟它们都是用来解决或大或小的逻辑问题或数学问题。STL收录的算法经过了数学上的效能分析与证明，是极具复用价值的，包括常用的排序，查找等等。特定的算法往往搭配特定的数据结构，算法与数据结构相辅相成。

算法分为:质变算法和非质变算法。

- 质变算法：是指运算过程中会更改区间内的元素的内容。例如拷贝，替换，删除等等
- 非质变算法：是指运算过程中不会更改区间内的元素内容，例如查找、计数、遍历、寻找极值等等

### 迭代器

迭代器(iterator)是一种抽象的设计概念，现实程序语言中并没有直接对应于这个概念的实物。 在《Design Patterns》一书中提供了23种设计模式的完整描述， 其中iterator模式定义如下：提供一种方法，使之能够依序寻访某个容器所含的各个元素，而又无需暴露该容器的内部表示方式。

迭代器的设计思维-STL的关键所在，STL的中心思想在于将容器(container)和算法(algorithms)分开，彼此独立设计，最后再一贴胶着剂将他们撮合在一起。

从技术角度来看，容器和算法的泛型化并不困难，c++的class template和function template可分别达到目标，如果设计出这两个之间的良好的胶着剂，才是大难题。

迭代器的种类:

| 迭代器         | 功能                                                         | 描述                                   |
| -------------- | ------------------------------------------------------------ | -------------------------------------- |
| 输入迭代器     | 提供对数据的只读访问                                         | 只读，支持++、==、!=                   |
| 输出迭代器     | 提供对数据的只写访问                                         | 只写，支持++                           |
| 前向迭代器     | 提供读写操作，并能向前推进迭代器                             | 读写，支持++、==、!=                   |
| 双向迭代器     | 提供读写操作，并能向前和向后推进迭代器                       | 读写，支持++、--                       |
| 随机访问迭代器 | 提供读写操作，并能以跳跃方式访问容器的任意数据，是功能最强的迭代器 | 读写，支持++、–、[n]、-n、<、<=、>、>= |

演示

```cpp
#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

//STL 中的容器 算法 迭代器
void test01(){
	vector<int> v; //STL 中的标准容器之一 ：动态数组
	v.push_back(1); //vector 容器提供的插入数据的方法
	v.push_back(5);
	v.push_back(3);
	v.push_back(7);
	//迭代器
	vector<int>::iterator pStart = v.begin(); //vector 容器提供了 begin()方法 返回指向第一个元素的迭代器
	vector<int>::iterator pEnd = v.end(); //vector 容器提供了 end()方法 返回指向最后一个元素下一个位置的迭代器
	//通过迭代器遍历
	while (pStart != pEnd){
		cout << *pStart << " ";
		pStart++;
	}
	cout << endl;
	//算法 count 算法 用于统计元素的个数
	int n = count(pStart, pEnd, 5);
	cout << "n:" << n << endl;
}
//STL 容器不单单可以存储基础数据类型，也可以存储类对象
class Teacher
{
public:
	Teacher(int age) :age(age){};
	~Teacher(){};
public:
	int age;
};
void test02(){
	vector<Teacher> v; //存储 Teacher 类型数据的容器
	Teacher t1(10), t2(20), t3(30);
	v.push_back(t1);
	v.push_back(t2);
	v.push_back(t3);
	vector<Teacher>::iterator pStart = v.begin();
	vector<Teacher>::iterator pEnd = v.end();
	//通过迭代器遍历
	while (pStart != pEnd){
		cout << pStart->age << " ";
		pStart++;
	}
	cout << endl;
}
//存储 Teacher 类型指针
void test03(){
	vector<Teacher*> v; //存储 Teacher 类型指针
	Teacher* t1 = new Teacher(10);
	Teacher* t2 = new Teacher(20);
	Teacher* t3 = new Teacher(30);
	v.push_back(t1);
	v.push_back(t2);
	v.push_back(t3);
	//拿到容器迭代器
	vector<Teacher*>::iterator pStart = v.begin();
	vector<Teacher*>::iterator pEnd = v.end();
	//通过迭代器遍历
	while (pStart != pEnd){
		cout << (*pStart)->age << " ";
		pStart++;
	}
	cout << endl;
}
//容器嵌套容器 难点
void test04()
{
	vector< vector<int> > v;
	vector<int>v1;
	vector<int>v2;
	vector<int>v3;

	for (int i = 0; i < 5;i++)
	{
		v1.push_back(i);
		v2.push_back(i * 10);
		v3.push_back(i * 100);
	}
	v.push_back(v1);
	v.push_back(v2);
	v.push_back(v3);

	for (vector< vector<int> >::iterator it = v.begin(); it != v.end();it++)
	{
		for (vector<int>::iterator subIt = (*it).begin(); subIt != (*it).end(); subIt ++)
		{
			cout << *subIt << " ";
		}
		cout << endl;
	}
} 
int main(){
	//test01();
	//test02();
	//test03();
	test04();
	system("pause");
	return EXIT_SUCCESS;
}

```



## 常用容器



### string容器

string容器基本概念
C风格字符串(以空字符结尾的字符数组)太过复杂难于掌握，不适合大程序的开发，所以C++标准库定义了一种string类，定义在头文件<string>。
String和c风格字符串对比：

- Char\*是一个指针，String是一个类 ；

  string封装了char\*，管理这个字符串，是一个char\*型的容器。

- String封装了很多实用的成员方法
  查找find，拷贝copy，删除delete 替换replace，插入insert

- 不用考虑内存释放和越界
  string管理char\*所分配的内存。每一次string的复制，取值都由string类负责维护，不用担心复制越界和取值越界等。

  

### string容器常用操作



**string 构造函数**

```cpp
string();//创建一个空的字符串 例如: string str;      
string(const string& str);//使用一个string对象初始化另一个string对象
string(const char* s);//使用字符串s初始化
string(int n, char c);//使用n个字符c初始化 
```

**string基本赋值操作**

```cpp
string& operator=(const char* s);//char*类型字符串 赋值给当前的字符串
string& operator=(const string &s);//把字符串s赋给当前的字符串
string& operator=(char c);//字符赋值给当前的字符串
string& assign(const char *s);//把字符串s赋给当前的字符串
string& assign(const char *s, int n);//把字符串s的前n个字符赋给当前的字符串
string& assign(const string &s);//把字符串s赋给当前字符串
string& assign(int n, char c);//用n个字符c赋给当前字符串
string& assign(const string &s, int start, int n);//将s从start开始n个字符赋值给字符串
```



**string存取字符操作**

```cpp
char& operator[](int n);//通过[]方式取字符
char& at(int n);//通过at方法获取字符
```

**string拼接操作**

```cpp
string& operator+=(const string& str);//重载+=操作符
string& operator+=(const char* str);//重载+=操作符
string& operator+=(const char c);//重载+=操作符
string& append(const char *s);//把字符串s连接到当前字符串结尾
string& append(const char *s, int n);//把字符串s的前n个字符连接到当前字符串结尾
string& append(const string &s);//同operator+=()
string& append(const string &s, int pos, int n);//把字符串s中从pos开始的n个字符连接到当前字符串结尾
string& append(int n, char c);//在当前字符串结尾添加n个字符c
```

**string查找和替换**

```cpp
int find(const string& str, int pos = 0) const; //查找str第一次出现位置,从pos开始查找
int find(const char* s, int pos = 0) const;  //查找s第一次出现位置,从pos开始查找
int find(const char* s, int pos, int n) const;  //从pos位置查找s的前n个字符第一次位置
int find(const char c, int pos = 0) const;  //查找字符c第一次出现位置
int rfind(const string& str, int pos = npos) const;//查找str最后一次位置,从pos开始查找
int rfind(const char* s, int pos = npos) const;//查找s最后一次出现位置,从pos开始查找
int rfind(const char* s, int pos, int n) const;//从pos查找s的前n个字符最后一次位置
int rfind(const char c, int pos = 0) const; //查找字符c最后一次出现位置
string& replace(int pos, int n, const string& str); //替换从pos开始n个字符为字符串str
string& replace(int pos, int n, const char* s); //替换从pos开始的n个字符为字符串s
```

**string比较操作**

```cpp
/*
compare函数在>时返回 1，<时返回 -1，==时返回 0。
比较区分大小写，比较时参考字典顺序，排越前面的越小。
大写的A比小写的a小。
*/
int compare(const string &s) const;//与字符串s比较
int compare(const char *s) const;//与字符串s比较
```

**string插入和删除操作**

```cpp
string& insert(int pos, const char* s); //插入字符串
string& insert(int pos, const string& str); //插入字符串
string& insert(int pos, int n, char c);//在指定位置插入n个字符c
string& erase(int pos, int n = npos);//删除从Pos开始的n个字符 
```





















