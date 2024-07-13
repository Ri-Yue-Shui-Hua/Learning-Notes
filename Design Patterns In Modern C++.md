## Design Patterns In Modern C++ 中文版翻译

[Design Patterns In Modern C++ 中文版翻译 - 《Design Patterns In Modern C++ 中文版翻译》 - 极客文档 (geekdaxue.co)](https://geekdaxue.co/read/design-pattern-in-modern-c/README.md)

### 动机



1. 本书的示例是用C++11、14、17和更高版本的现代C++编写的，有助于熟悉现代C++的语法。
2. 设计模式是编程经验的总结，广泛存在于工程实践中，牵扯出非常多的相关内容（比大家熟悉的单例模式为例，可以引出C++11后的多线程内存模型，除了用局部静态变量还可以用Acquire and Release栅栏， Sequentially Consistent 原子操作等无锁方式实现，以及folly中如何在工业实践中实现Singleton来管理多个Singletons），以此为线索梳理所学的知识。
3. 打算在原书的基础上补充大量的相关知识，如STL、Boost和folly中的设计模式，举例Leetcode题目中的设计模式，还有融入多线程并发情况下的一些例子。

### TODO

-  Chapter01: Introduction. 我直接使用了[@soyoo的翻译结果](https://github.com/soyoo/design_patterns_in_modern_cpp_zh_sample)
-  Chapter02: Builder。
-  Chapter03: Factories. 涉及工厂方法、工厂、内部工厂、抽象工厂和函数工厂。
-  Chapter04: Prototype. 原型模式，对深拷贝的实现做了一些讨论(拷贝构造函数和拷贝复制运算符；序列化和反序列化)。
-  Chapter05: Singleton. 线程安全，单例的问题，控制反转和Monostate.
-  Chapter06: Adapter. 额外补充了STL中queue的实现，提供了一个更安全和方法的Queue。需要了解boost库中的hash是怎么做的。
-  Chapter07: Bridge. 增加了Pimpl编程技法的说明。
-  Chapter08: Composite.
-  Chapter09：Decorator. 涉及动态装饰器、静态装饰器 和 函数装饰器。
-  Chapter10: Facade. 外观模式, 缓冲-视窗-控制台。
-  Chapter11: Flyweight. 享元模式。Boost库中Flyweight的实现，以及Bimap
-  Chapter12: Proxy. 智能指针、属性代理、虚代理和通信代理。
-  Chapter13: Chain of Responsibility. 指针链；代理链涉及中介模式和观察者模式。
-  Chapter14: Command.
-  Chapter15: Interpreter.涉及编译原理里面的词法分析，语法分析，`Boost.spirit`的使用。后面会补充LeetCode上实现计算器的几道题目和正则表达式的题目，也许会增加`Lex/Yacc`工具的使用介绍，以及tinySQL解释器实现的简单解释。
-  Chapter16: Iterator. STL库中的迭代器，涉及二叉树的迭代器，使用协程来简化迭代过程。
-  Chapter17: Mediator.
-  Chapter18: Memento.
-  Chapter19: Nulll Object. 涉及到对代理模式和pimpl编程技法的运用，以及std::optional
-  Chapter20: Observer. 属性观察者、模板观察者Observer\、可观察Observable\ 、依赖问题, 取消订阅与线程安全 和使用Boost.Signals2 来实现 Observer。
-  Chapter21: State. 补充字符串匹配、例子
-  Chapter22: Strategy. 动态策略和静态策略
-  Chapter23: Template Method. 模版方法模式和策略模式的异同。
-  Chapter24: Visitor. 入侵式、反射式、经典式的访问者的设计思路，std::visitor在variant类型上的访问。

### 补充

在原著的基础上补充了很多相关的东西，

- 第5章-单例：补充了无锁的double-check实现。
- 第6章-适配器：探讨了STL中queue的适配器设计，提供了一个更方便和更安全的`Queue`适配器实现。
- 第7章-桥接：对C++中的Pimpl编程技法进行了补充，给出了在编译器方面的应用。
- 第12章-代理：讨论C++中智能指针的实现，给出一个半线程安全的智能指针`shared_ptr`的实现。
- 第20章-观察者：补充了由观察者模式衍生出来的发布-订阅模式，总结了消息队列使用注意事项，提供了2种用redis来实现消息队列的解决方案。

#### **第5章-单例：无锁的double-check实现**



```cpp
class Singleton
{
    protected:
        Singleton();
    private:
        static std::mutex m_mutex;
        static std::atomic<Singleton*> m_instance = nullptr;
    public:
        static Singleton* Singleton::getInstance() 
        {
            Singleton* tmp = m_instance.load(std::memory_order_acquire);
            if (tmp == nullptr) 
            {
                //std::scoped_lock(m_mutex);
                std::lock_guard<std::mutex> lock(m_mutex);
                tmp = m_instance.load(std::memory_order_relaxed);
                if (tmp == nullptr) 
                {
                    tmp = new Singleton;
                    m_instance.store(tmp, std::memory_order_release);
                }
            }
            return tmp;
        }
        Singleton(const Singleton&) = delete;
        Singleton& operator=(const Singleton&) = delete;
        Singleton(Singleton&&) = delete;
        Singleton& operator=(Singleton&&) = delete;
};
```

#### **第6章-适配器：设计更安全方便的Queue**

STL中`queue`是一个FIFO队列，提供的核心接口函数为

- push( ) : 插入一个元素到队列中
- front( ) : 返回队首元素
- back( ) : 返回队尾元素
- pop( ) : 移除队首元素

我们可以看到在STL头文件`<queue>`中`queue`中定义：

```cpp
namespace std
{
    template<typename T, typename Container = deque<T> >
    class queue;
}
```

注意到第二个可选参数，说明在`queue`中默认使用`deuqe`作为实际存储`T`类的容器, 当然也可以使用任何提供成员函数`front()`、`back()`、`push_back()`和`pop_front()`的序列容器类，如`list`。标准库中的`queue`的实现更注重效率而不是方便和安全。这并不是适合于所有场合。我们可以在一个提供`deque`容器做出修改，适配出一个不同于标准库但符合自己风格的`Queue`。

下面实现的Queue提供了抛出异常的处理，以及返回队首元素的新的`pop`方法。

```cpp
template <typename T,  Container = deque<T> >
class Queue
{
    protected:
        Container c;
        //异常类：在一个空队列中调用 pop() 和 front() 
        class EmptyQueue : public exception
        {
            public:
                virtual const char* what const throw( )
                {
                    return "empty queue!";
                }
        }
        typename Container::size_type size( ) const
        {
            return c.size( );
        }
        bool empty( ) bool 
        {
            return c.empty();
        }
        void push( const T& item )
        {
            c.push( item );
        }
        void push( T&& item )
        {
            c.push( item );
        }
        const T& front( ) const
        {
            if( c.empty( ) )
                throw EmptyQueue();
            return c.front();
        }
        T& front( )
        {
             if( c.empty( ) )
                throw EmptyQueue( );
            return c.front( );
        }
        const T& back( ) const
        {
            if( c.empty( ) )
                throw EmptyQueue();
            return c.back();
        }
        T& front( )
        {
             if( c.empty( ) )
                throw EmptyQueue( );
            return c.back( );
        }
        // 返回队首元素
        T pop( ) const 
        {
            if( c.empty() )
                throw EmptyQueue();
            T elem( c.front( ) );
            c.pop();
            return elem;
        }
};
```

#### 第7章-桥接：Pimpl编程技法-减少编译依赖



PImpl（Pointer to implementation）是一种C++编程技术，其通过将类的实现的详细信息放在另一个单独的类中，并通过不透明的指针来访问。这项技术能够将实现的细节从其对象中去除，还能减少编译依赖。有人将其称为“编译防火墙（Compilation Firewalls）”。

##### Pimpl技法的定义和用处

在C ++中，如果头文件类定义中的任何内容发生更改，则必须重新编译该类的所有用户-即使唯一的更改是该类用户甚至无法访问的私有类成员。这是因为C ++的构建模型基于文本包含（textual inclusion），并且因为C ++假定调用者知道一个类的两个主要方面，而这两个可能会受到私有成员的影响：

- 因为类的私有数据成员参与其对象表示，影响大小和布局，
- 也因为类的私有成员函数参与重载决议（这发生于成员访问检查之前），故对实现细节的任何更改都要求该类的所有用户重编译。

为了减少这些编译依赖性，一种常见的技术是使用不透明的指针来隐藏一些实现细节。这是基本概念：

```cpp
// Pimpl idiom - basic idea
class widget {
    // :::
private:
    struct impl;        // things to be hidden go here
    impl* pimpl_;       // opaque pointer to forward-declared class
};
```

类widget使用了handle/body编程技法的变体。handle/body主要用于对一个共享实现的引用计数，但是它也具有更一般的实现隐藏用法。为了方便起见，从现在开始，我将widget称为“可见类”，将impl称为“ Pimpl类”。

编程技法的一大优势是，它打破了编译时的依赖性。首先，系统构建运行得更快，因为使用Pimpl可以消除额外的#include。我从事过一些项目，在这些项目中，仅将几个广为可见的类转换为使用Pimpls即可使系统的构建时间减少一半。其次，它可以本地化代码更改的构建影响，因为可以自由更改驻留在Pimpl中的类的各个部分，也就是可以自由添加或删除成员，而无需重新编译客户端代码。由于它非常擅长消除仅由于现在隐藏的成员的更改而导致的编译级联，因此通常被称为“编译防火墙”。

##### Pimpl技法的实践

避免使用原生指针和显式的`delete`。要仅使用C++标准设施表达`Pimpl`，最合适的选择是通过`unique_ptr`来保存`Pimpl`对象，因为Pimpl对象唯一被可见类拥有。使用`unique_ptr`的代码很简单：

```cpp
// in header file
class widget {
public:
    widget();
    ~widget();
private:
    class impl;
    unique_ptr<impl> pimpl;
};
// in implementation file
class widget::impl {
    // :::
};
widget::widget() : pimpl{ new impl{ /*...*/ } } { }
widget::~widget() { }                   // or =default
```

//TODO: handle/body编程技法

#### 第12章-代理：实现一个半线程安全的智能指针

1. 智能指针(shared_ptr)线程安全吗?

- （”half thread-safe”）
- 是: 引用计数控制单元线程安全, 保证对象只被释放一次
- 否: 对于数据的读写没有线程安全

1. 如何将智能指针变成线程安全?

- 使用 mutex 控制智能指针的访问
- 使用全局非成员原子操作函数访问, 诸如: std::atomic_load(), atomic_store(), …
- 缺点: 容易出错, 忘记使用这些操作
- C++20: atomic>, atomic> std::atomic_shared_ptrand astd::atomic_weak_ptr.
- 内部原理可能使用了 mutex
- 全局非成员原子操作函数标记为不推荐使用(deprecated)

1. 数据竞争

- 一个shared_ptr对象实体可以被多个线程同时读取
- 两个shared_ptr对象实体可以被两个线程同时写入，”析构”算写操作
- 如果要从多个线程读写同一个shared_ptr,那么需要加锁。

1. 代码实现

```cpp
#include <atomic>
// 非完全线程安全的。
// 引用计数
template<typename T>
class ReferenceCounter
{
    ReferenceCounter( ):count(0) {};
    ReferenceCounter(const ReferenceCounter&) = delete;
    ReferenceCounter& operator=(const ReferenceCounter&) = delete;
    ReferenceCounter(ReferenceCounter&&) = default
    ReferenceCounter& operator=(ReferenceCounter&&) = default;
    // 前置++, 这里不提供后置，因为后置返回一个ReferenceCounter的拷贝，而之前禁止ReferenceCounter拷贝。
    ReferenceCounter& operator++()
    {   
        count.fetch_add(1);
        return *this;
    }
    ReferenceCounter& operator--()
    {
        count.fetch_sub(1);
        return *this;
    }
    size_t getCount() const 
    {
        return count.load();
    }
    private:
        // 原子类型，或上锁
        std::atomic<size_t> count;
};
template <typename T>
class SharedPtr{
    explicit SharedPtr(T* ptr_ = nullptr) : ptr(ptr_) {
        count = new ReferenceCounter();
        if(ptr)
        {
            ++(*count);
        } 
    }
    ~SharedPtr() {
        --(*count);
        if(count->getCount() == 0)
        {
            delete ptr;
            delete count;
        }
    }
    SharedPtr(const SharedPtr<T>& other) : 
        ptr(other.ptr), 
        count(other.count)
    {
        ++(*count);
    }
    SharedPtr<T>& operator= (const SharedPtr<T>& other)
    {
        if(this != &other)
        {
            --(*count);
            if(count.getCount() == 0)
            {
                delete count;
                delete ptr;
            }
            ptr = other.ptr;
            count = other.count;
            ++(*count);
        }
        return *this;   
    }
    SharedPtr(SharedPtr&& other) = default;
    SharedPtr& operator=(SharedPtr&& other) = default;
    T& operator*() const { return *ptr; }
    T* operator->() const{
        return ptr;
    }
    T* get() const { return ptr; }
    int use_count() const { return count->getCount(); }
    bool unique() const { return use_count() == 1; }
    private:
        T* ptr;
        ReferenceCounter* count;
}
```



1. 重新回顾C++线程池中使用的虚调用方法

#### 第20章：观察者模式

从观察者模式出发，了解了发布订阅模式，到消息队列，再到用redis实现消息队列以及实现消息队列的注意事项

##### 观察者模式

##### 发布订阅模式

- 发布者和订阅者不直接关联，借助消息队列实现了松耦合，降低了复杂度，同时提高了系统的可伸缩性和可扩展性。
- 异步

##### 消息队列

##### 使用redis实现消息队列

- 如何保证有序：FIFO数据结构
- 如何保证不重复：全局Id(LIST法1：生产者和消费者约定好；STREAMS法二：消息队列自动产生（时间戳）)
- 如何保证可靠性：备份

| 功能和适用场景 | 基于List                           | 基于Streams                                                  | 备注                                                         |
| :------------- | :--------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| 适用场景       | Redis5.0版本的部署环境，消息总量小 | Redis5.0及以后的版本的部署环境，消息总量大，需要消费组形式读取数据 |                                                              |
| 消息保序       | LPUSH/RPOP                         | XADD/XREAD                                                   |                                                              |
| 阻塞读取       | BRPOP                              | XREAD block                                                  | 阻塞式读取在客户端没有读到队列数据时，自动阻塞，节省因不断peek的CPU开销 |
| 重复消息处理   | 生产者自行实现全局唯一ID           | STREAMs自行生成全局唯一ID                                    |                                                              |
| 消息可靠性     | 适用BRPOPRPUSH                     | 使用PENDING List 自动存留消息，使用XPENGDING查看，使用XACK确认消息 |                                                              |



##### 消息队列的注意事项

在使用消息队列时，重点需要关注的是如何保证不丢消息？

那么下面就来分析一下，哪些情况下，会丢消息，以及如何解决？

1、生产者在发布消息时异常：

a) 网络故障或其他问题导致发布失败（直接返回错误，消息根本没发出去） b) 网络抖动导致发布超时（可能发送数据包成功，但读取响应结果超时了，不知道结果如何）

情况a还好，消息根本没发出去，那么重新发一次就好了。但是情况b没办法知道到底有没有发布成功，所以也只能再发一次。所以这两种情况，生产者都需要重新发布消息，直到成功为止（一般设定一个最大重试次数，超过最大次数依旧失败的需要报警处理）。这就会导致消费者可能会收到重复消息的问题，所以消费者需要保证在收到重复消息时，依旧能保证业务的正确性（设计幂等逻辑），一般需要根据具体业务来做，例如使用消息的唯一ID，或者版本号配合业务逻辑来处理。

2、消费者在处理消息时异常：

也就是消费者把消息拿出来了，但是还没处理完，消费者就挂了。这种情况，需要消费者恢复时，依旧能处理之前没有消费成功的消息。使用List当作队列时，也就是利用老师文章所讲的备份队列来保证，代价是增加了维护这个备份队列的成本。而Streams则是采用ack的方式，消费成功后告知中间件，这种方式处理起来更优雅，成熟的队列中间件例如RabbitMQ、Kafka都是采用这种方式来保证消费者不丢消息的。

3、消息队列中间件丢失消息

上面2个层面都比较好处理，只要客户端和服务端配合好，就能保证生产者和消费者都不丢消息。但是，如果消息队列中间件本身就不可靠，也有可能会丢失消息，毕竟生产者和消费这都依赖它，如果它不可靠，那么生产者和消费者无论怎么做，都无法保证数据不丢失。

a) 在用Redis当作队列或存储数据时，是有可能丢失数据的：一个场景是，如果打开AOF并且是每秒写盘，因为这个写盘过程是异步的，Redis宕机时会丢失1秒的数据。而如果AOF改为同步写盘，那么写入性能会下降。另一个场景是，如果采用主从集群，如果写入量比较大，从库同步存在延迟，此时进行主从切换，也存在丢失数据的可能（从库还未同步完成主库发来的数据就被提成主库）。总的来说，Redis不保证严格的数据完整性和主从切换时的一致性。我们在使用Redis时需要注意。

b) 而采用RabbitMQ和Kafka这些专业的队列中间件时，就没有这个问题了。这些组件一般是部署一个集群，生产者在发布消息时，队列中间件一般会采用写多个节点+预写磁盘的方式保证消息的完整性，即便其中一个节点挂了，也能保证集群的数据不丢失。当然，为了做到这些，方案肯定比Redis设计的要复杂（毕竟是专们针对队列场景设计的）。

综上，Redis可以用作队列，而且性能很高，部署维护也很轻量，但缺点是无法严格保数据的完整性（个人认为这就是业界有争议要不要使用Redis当作队列的地方）。而使用专业的队列中间件，可以严格保证数据的完整性，但缺点是，部署维护成本高，用起来比较重。

所以我们需要根据具体情况进行选择，如果对于丢数据不敏感的业务，例如发短信、发通知的场景，可以采用Redis作队列。如果是金融相关的业务场景，例如交易、支付这类，建议还是使用专业的队列中间件。

##### 消息队列实现方式对比

关于Redis是否适合做消息队列，业界一直存在争议。很多人认为使用消息队列就应该采用KafKa，RabbitMQ这些专门没安心消息队列场景的软件，而Redis更适合做缓存。

| 功能和适用场景 | 基于Redis                                | 基于Kafka/RabbitMQ                                           | 备注 |
| :------------- | :--------------------------------------- | :----------------------------------------------------------- | :--- |
| 适用场景       | 对丢数据不敏感的业务，如发短信和通知     | 严格数据完整的应用，金融相关业务场景，支付和交易             |      |
| 属性           | 轻量级，部署和维护简单，性能高           | 重量级                                                       |      |
| 数据完整性     | 不保证严格的数据完整性和主从切换的一致性 | 部署集群，多结点+预写磁盘保证数据完整性                      |      |
| 重复消息处理   | 生产者自行实现全局唯一ID                 | STREAMs自行生成全局唯一ID                                    |      |
| 消息可靠性     | 适用BRPOPRPUSH                           | 使用PENDING List 自动存留消息，使用XPENGDING查看，使用XACK确认消息 |      |

### 如何学习设计模式(即如何学习C++的面向对象思想)

[该内容来自@franktea](https://github.com/franktea)

#### 面向对象的三大特征

所有的人都知道，面向对象的三大特征是封装继承多态（这里的多态指的是运行时多态，本文中所有多态均指运行时多态），那么哪个特征才是面向对象最重要的特征呢？

先说封装。其实C语言的struct也是一种封装，所以封装并不是面向对象所独有。再看继承，继承可以非常方便地重用代码，相对面向过程来说是一种非常强大的功能，在面向对象刚被发明出来不久的一段时间里，继承被很多人看成面向对象最强大的特征。

到后来，人们发现面向对象最强大的特征是多态，因为代码不仅仅是需要重用，扩展也很重要。「设计模式」中，几乎每种模式，都是用多态来实现的。

一个问题：只支持多态，不支持继承的编程语言，算是面向对象的编程语言吗？

我的答案：不是。虽然继承不如多态重要，但是它不是多余的。多态往往是配合继承才更强大。

#### 类的设计以及多态

第一点，前面说了，面向对象最重要的是多态，多态就是使用虚函数，在自己设计的类中，将哪些成员函数定义为虚函数，这是一个重要的问题。对于新手，我的建议是：在搞清规则之前，可以将所有的成员函数都定义为虚函数。（其实在java这样的编程语言中，根本不需要程序员自己去指定哪个成员函数是virtual，从语法上来说，任何一个非static非private的都是virtual。）

在虚函数的定义上，先将所有能定义成虚函数的的成员函数全部声明为virtual，然后再在使用中慢慢做减法，根据自己的理解，将多余的virtual去掉。

第二点，在使用面向对象的时候，尽量使用父类指针，而不是子类指针。100分的设计是永远使用父类指针、永远不使用子类指针。父类指针向子类指针转换需要用dynamic_cast，不使用dynamic_cast的设计就是最好的设计。新手可以遵循这个原则。

当然，在一些非常复杂的系统中，无法做到100分，有时候还是需要向下转换成子类指针，这样的设计肯定是扣分的，但是对于复杂系统肯定有一个平衡。我自己做服务器，所有设计都可以做到永远使用父类指针，但是对于复杂的像客户端unnreal代码，向下转换几乎不可避免。

#### 虚函数表和虚函数指针

关于面向对象的语法知识，我唯一觉得需要强调的就是对于vtable的实现，推荐大家用实验的方法，一定要自己写代码亲自操作一遍（按照随便一篇vtable原理的文章动手操作一遍即可），无论是通过单步调试去查看vtable，还是通过编译器的各种命令来查看，都要自己亲自动手操作一下，加深印象。

#### 补充

gdb调试要点： 64位, 设置断点，打印虚表。

```bash
g++ 多继承有虚函数重写.cpp -o 多继承有虚函数重写 -m64 -g
break 30
set print pretty on
info vtbl d
```

### 一篇引文：从vtable到设计模式——我的C++面向对象学习心得

[该内容来自@franktea](https://github.com/franktea)

#### 前言

按照很多教程的内容安排，学习C++语法以后很快就会进入到面向对象的学习，在初学者的心中，面向对象有非常重要的地位。但是如何才能快速学习面向对象、多久学会面向对象才算正常，这是新手常见的问题。

面向对象的语法书上都有说，vtable的原理也有非常多的文章进行讲述，这些东西再一再重复没有意义，我想写一些我自己在学习过程中的心得体验。

关于面向对象的语法知识，我唯一觉得需要强调的就是对于vtable的实现，推荐大家用实验的方法，一定要自己写代码亲自操作一遍（按照随便一篇vtable原理的文章动手操作一遍即可），无论是通过单步调试去查看vtable，还是通过编译器的各种命令来查看，都要自己亲自动手操作一下，加深印象。

面向对象的语法风格出现以后，无数的程序员基于这些特性写出了很多代码，各显神通，后来被总结提取出一些可复用的方法论，叫做「设计模式」。设计模式是学习和掌握面向对象思想的重要课程。那么问题来了，何时学习设计模式？如何学习？在理解设计模式之前应该做什么、能做什么？

#### 封装、继承、（运行时）多态，哪个才是面向对象最重要的特征

所有的人都知道，面向对象的三大特征是封装继承多态（这里的多态指的是运行时多态，本文中所有多态均指运行时多态），那么哪个特征才是面向对象最重要的特征呢？

先说封装。其实C语言的struct也是一种封装，所以封装并不是面向对象所独有。再看继承，继承可以非常方便地重用代码，相对面向过程来说是一种非常强大的功能，在面向对象刚被发明出来不久的一段时间里，继承被很多人看成面向对象最强大的特征。

到后来，人们发现面向对象最强大的特征是多态，因为代码不仅仅是需要重用，扩展也很重要。「设计模式」中，几乎每种模式，都是用多态来实现的。

一个问题：只支持多态，不支持继承的编程语言，算是面向对象的编程语言吗？

我的答案：不是。虽然继承不如多态重要，但是它不是多余的。多态往往是配合继承才更强大。

#### 设计模式的意义

设计模式对于如何用面向对象的思想解决软件中的设计和实现问题提供了一些可重用的思路，它还有一个重要的意义，就是为每种设计思路都取了名字，便于程序员之间的交流。

有些人在设计类的名字的时候就包含了使用的设计模式，比如一个使用了adapter模式的类名字叫xxxAdapter；xxxFactory一看就知道它使用了factory模式，给其它使用和维护这些代码的人节省了大量的时间。

#### 何时开始学习设计模式

知乎上见过一个问题：『你想对刚毕业的人说些什么』，这个问题就是一个刚踏入社会的小鲜肉，向在社会上摸爬滚打多年的人取经，想获得一些生存闯关的金句宝典，从而让自己少踩坑。

这样的问题的答案有意义吗？有一些是有的，可以直接理解，但是很多是要结合自己过去的经验教训才能有体会的，知道得早也没有什么收获。

如果面向对象的初学者也提问：「你想对刚学习面向对象的人说什么？」，答案就在设计模式这本书中。

所以何时开始学习设计模式呢？我的答案是任何时候都可以。但是唯一要注意的就是，不要强迫自己去理解，设计模式的书可以摆在那里，想看就看一下，能理解多少就理解多少。但是越早看设计模式这本书，共鸣就越少，因为共鸣是要结合自己写面向对象代码的经验的。

学习面向对象几年以后再看设计模式是否可以行？

我觉得可行，结合自己几年之内在学习各种面向对象的库和自己写代码的经验，学习设计模式会很快。

永远不学设计模式行不行？

我觉得不行，我前面提到了，设计模式不仅仅是总结思想，思想可以通过模仿现有的库来学习，但是设计模式还有一个重要的作用是给模式命名，命名可以更好地与其他程序员沟通交流。

#### 如何学习设计模式(即如何学习C++的面向对象思想)

除了学习C++语法，还需要学习一下UML类图，不会的自己去搜，UML有好几种图，其中类图、状态图、序列图最为常用，学这3种即可。

在C++中可以通过学Qt库来学习面向对象。Qt除了可以用来写跨平台的UI，还可以写一些简单的网络程序，在学校里可以用来做各种大作业，无论是学生成绩管理系统、图书管理系统、足球俱乐部，等等，用Qt都可以很好地完成。我学Qt用的是这本书：<>

Qt里面本身就用了很设计模式，从它的类里面继承一个子类，覆盖一个或几个虚函数，就可以将自己的类融入到Qt的体系中。其实这就是学习面向对象的第一步，也是最好的开始，不吃猪肉、先看猪跑，从它的类继承多了，自己也会慢慢理解如何从自己写的类继承。

学习面向对象有什么减少弯路又能加速理解的套路呢？根据我自己的经验总结，对于新手我至少可以说两点。

第一点，前面说了，面向对象最重要的是多态，多态就是使用虚函数，在自己设计的类中，将哪些成员函数定义为虚函数，这是一个重要的问题。对于新手，我的建议是：在搞清规则之前，可以将所有的成员函数都定义为虚函数。（其实在java这样的编程语言中，根本不需要程序员自己去指定哪个成员函数是virtual，从语法上来说，任何一个非static非private的都是virtual。）

在虚函数的定义上，先将所有能定义成虚函数的的成员函数全部声明为virtual，然后再在使用中慢慢做减法，根据自己的理解，将多余的virtual去掉。

第二点，在使用面向对象的时候，尽量使用父类指针，而不是子类指针。100分的设计是永远使用父类指针、永远不使用子类指针。父类指针向子类指针转换需要用dynamic_cast，不使用dynamic_cast的设计就是最好的设计。新手可以遵循这个原则。

当然，在一些非常复杂的系统中，无法做到100分，有时候还是需要向下转换成子类指针，这样的设计肯定是扣分的，但是对于复杂系统肯定有一个平衡。我自己做服务器，所有设计都可以做到永远使用父类指针，但是对于复杂的像客户端unnreal代码，向下转换几乎不可避免。

#### 多久学会设计模式才算优秀

初学者都很急于求成，希望一天就能学会。但是从另一个角度来说，一天都能学会的东西，肯定不是什么有价值的东西。

我大概用了4年左右的时间，理解了面向对象。从大一开始学习C++，到大四毕业工作以后一年内设计出来了一个总共有一千多个类的系统，可以按照需求无限扩展。我现在可以设计任意多个类的系统。

我相信很多人比我更优秀，但是我更相信我自己的方法，我的学习方法其实就是不给自己设置时间期限，盲人摸象，今天摸这里明天摸那里，时间长了总会知道大象的全貌。

我是打算用几十年的时间从事编程的工作，到底是一天理解还是几年理解，对我来说并没有区别。至于做题、考试、工作等等，不用理解一样可以完成，按照现有的系统模仿即可。

我很清楚地记得，我第一次体会到面向对象的意义，是模仿MFC的一个机制。MFC在90年代的时候就做到了可以用字符串来动态创建一个对象（C++没有反射机制这在语法层面是无法做到的），MFC用的方法非常简单，将所有的类的名字和其构造函数放在一个全局的链表中，通过字符串在链表中去查找对应的构造函数，从而调用该构造函数new出对应的对象。

需要添加新的功能的时候，只要新添加一个.h一个.cpp，在两个文件中实现一个子类的代码，并调用宏将该类的构造函数添加到全局链表中。

通过添加新文件（一个.h和一个.cpp）的方法，不用修改之前的任何代码，就扩展了程序的功能，这就是面向对象的意义之一。

后来我在鹅厂做服务器，这个方法我一直使用，只是将链表改成了map或unordered_map。以后如果我找到合适的例子，我想通过例子说明此思想的应用，作为面向对象思想理解的入门级素材，我觉得挺好的，当然，那就是另外一篇文章了。

#### 总结

设计模式是一些方法论，自己通过学习优秀的C++框架（如Qt）慢慢去体会和应用这些方法，最终可以慢慢理解。

不要刻意急于求成，人生很长，每一步都有它的意义，走过的路哪怕是弯路，都有它的意义。

在理解之前，注重于模仿，即使不理解，靠模仿已经能解决很多问题。

如果硬要问捷径是什么，我的答案就是抓紧时间多写代码，写了几万行代码就慢慢理解了。如果你不能改变几万行这个数字，那就去改变积累几万行代码的时间。比如说从3年缩短到2年，这完全是可能的。

我非常讨厌写长文，这篇文章在没有任何代码凑字数的情况下还是超过了3000字，也是源于我对面向对象思想的热爱，它帮我解决了很多问题，我现在用面向对象的思想来写代码，已经成了一件很自然的事情。

其实面向对象的思想在C++中并不是主流，自从90年代STL被作为标准库纳入C++那一刻起，泛型编程在C++里面就占据了上风，并且后来一直在迅速发展。同样的设计模式在C++中不仅仅可以用面向对象的思想实现，也可以用泛型编程的思想实现，不少时候后者可能更神奇更优雅更高效。

面向对象注重的是代码的扩展和维护，而不是高性能，在一些需要高性能的场合，像我所在的游戏领域需要优化性能的地方，不能用面向对象，以后如果我找到合适的例子作为素材，我会再写一篇「面向对象的缺点」的文章。

### 声明

译者纯粹出于学习目的与个人兴趣翻译本书，不追求任何经济利益。

本译文只供学习研究参考之用，不得公开传播发行或用于商业用途。有能力阅读英文书籍者请购买正版支持。

### 许可

[CC-BY 4.0](https://geekdaxue.co/read/design-pattern-in-modern-c/$LICENSE)

### 参考

1. [《Design Patterns In Modern C++》](https://book.douban.com/subject/30200080/)
2. [《The C++ Standard Library - A Tutorial and Reference, 2nd Edition》](http://cppstdlib.com/)





## 第一章 介绍



“设计模式” 这个话题听起来很枯燥，从学术上来说，又略显呆滞；老实说，任何能够想象到的编程语言都把这个讲烂了——包括一些编程语言，例如 JavaScript，甚至没有正确的面向对象编程特征！那为什么还要写另一本书呢？

我想这本书存在的主要原因，那就是 C++ 又再次焕发生机。在经历了很长一段时间的停滞之后，它正在进化、成长，尽管它又不得不向后与 C 语言兼容做斗争，但是，好的事情正在发生，尽管不是我们所预期的速度（例如 modules，还有其它东西）。

现在，关于设计模式，我们不应该忘记最初出版的设计模式书籍[^1]，其中的示例是使用 C++ 与 Smalltalk写就的。从那时起，许多编程语言都将设计模式直接融入到语言当中：例如，C# 直接将观察者模式与其对事件的内置支持结合在一起（对应于 event 关键字）。C++ 没有这样实现，至少在语法级别上没有这样做。尽管如此，像诸如 std::function 这样的特性的引入确实使许多编程场景变得更加简单。

注1：Erich Gamma et al., *Design Patterns: Elements of Reusable Object-Oriented Software* (Boston, MA: Addison Wesley, 1994).

同时，设计模式也是一项有趣的研究，比如，如何通过不同复杂程度的技术，不同类型的权衡，来解决一个问题。有些设计模式或多或少是必要的、不可避免的，而其它设计模式更多的是科学上的求知欲（尽管如此，这本书还是会讨论的，因为我是一个完美主义者）。

读者应该意识到，对于某些问题的综合解决方案（例如，观察者模式）通常会导致过度设计；也就是说，创建比大多数典型场景所需的复杂得多的结构。虽然，过度设计具有很多乐趣（嘿嘿，你能真正解决问题，并给同事留下深刻印象），但这往往是不可行的。

### 预备知识

#### 这本书是为谁写的

这本书被设计的更现代，是对经典 GoF 书的更新的，尤其针对 C++ 编程语言。我的意识是，你们中还有多少人在写 Smalltalk？不是很多，这是我的猜测。

这本书的目的是研究如何将现代 C++（目前，可用的 C++ 最新版本）应用于经典设计模式的实现。与此同时，它也是尝试实现任何新模式、新方法，只要是有利于 C++ 开发人员的。

最后，在一些地方，这本书只是现代 C++ 的一个十分简单的技术演示，展示了它的一些最新特性（例如，coroutines）是如何使难题变得更容易解决。

#### 代码示例

本书中的示例都适合于投入到生产环境中，但是，为了便于阅读，我们做了一些简化：

- 经常的，你会发现我使用 struct 去替代 class，仅仅为了避免在太多的地方书写 public 关键字。
- 我将避免使用 std:: 前缀，因为它会损害可读性，特别是在代码密度很高的地方。如果我使用了 string，你可以打赌我指的是 std::string。
- 我将避免添加虚析构函数，而在现实生活中，添加它们可能是有意义的。
- 在极少数情况下，我将按值创建、传递参数，以避免 shared_ptr、 make_shared 等等的扩散。智能指针增加了另外一个层次的复杂度，将它们集成到本书中介绍的设计模式中，作为一个练习留给读者。
- 我有时会省略一些代码元素，它们对于完成一个类型的功能是必要的（例如，移动构造函数），因为这些元素占用了太多的空间。
- 在很多情况下，我会忽略 const 关键字；在正常情况下，这实际上是有意义的。const 正确性通常会导致 api 表面上的分裂、加倍，这在书的格式中不能很好地工作。

你应该意识到，大多数示例都使用了现代 C++（C++11、14、17和更高版本），并且，开发人员通常可以使用最新的 C++ 语言特性。例如，当 C++14 允许我们自动推断返回值类型时，你将不会发现许多函数签名以 -> decltype(…) 为结尾。这些示例中没有一个是针对特定编译器的，但是如果你选择的编译器[^2]不能正常工作，你需要找到解决办法。

注2：Intel, I’m looking at you!

在某些情况下，我将引用其它编程语言，比如 C# 或者 Kotlin。有时值得注意的是，其它语言设计者是如何实现特定功能的？对于 C++ 来说，从其它语言借鉴一般可用的想法并不陌生：例如，在许多其它语言中，引入了 auto 关键字用于变量声明和返回类型的自动推断。

##### 开发工具

本书中编写的代码示例是用于现代 C++ 编译器的，像 Clang，GCC 或者 MSVC。我一般假设你使用的是可用的最新编译器版本，因此，将使用我可以使用的最新、最优秀的语言特性。在某些情况下，高级语言示例对于早期编译器需要降级使用；而在其它情况下，则可能无法实现。

就开发人员的工具而言，这本书没有具体涉及到它们，因此，如果你有一个最新的编译器，你应该很好地遵循这些示例：它们中的大多数都是自包含的 .cpp 文件。尽管如此，我还是想借此机会提醒你，诸如 CLion 或 ReSharper C++ 之类的质量开发人员工具极大地提高了开发体验。只要你投资一小笔钱，你就可以获得大量的额外功能，这些功能可以直接转化为编码速度和代码质量的提高。

##### 盗版

数字盗版是一个不可逃避的事实。一个崭新的一代正在成长，从来没有购买过一部电影或一本书籍，甚至这本书。这也没什么可做的。我唯一能说的是，如果你翻版这本书，你可能不会读最新的版本。

在线数字出版的乐趣在于，我可以把这本书更新为 C++ 的最新版本，我也可以做更多的研究。因此，如果你为这本书付费，当 C++ 语言和标准库的新版本发布时，你将在将来获得免费的更新。如果不付费，哦，好吧…

#### 重要的概念

在我们开始之前，我想简单地提及在这本书中将要引用的 C++ 世界的一些关键概念。

##### 奇异递归模板模式（CRTP）

嗨，很显然，这是一个模式！我不知道它是否有资格被列为一个独立的设计模式，但是，它肯定是 C++ 世界中的一个模式。从本质上说，这个想法很简单：继承者将自己作为模板参数传递给它的基类：

```cpp
struct Foo : SomeBase<Foo>
{
    ...
}
```

现在，您可能想知道为什么有人会这么做？原因之一是，以便于能够访问基类实现中的类型化 this 指针。

例如，假设 SomeBase 的每个继承者都实现了迭代所需的 begin()/end() 对。那么，您将如何在 SomeBase 的成员中迭代该对象？直觉表明，您不能这样做，因为 SomeBase 本身没有提供 begin()/end() 接口。但是，如果您使用 CRTP，实际上是可以将 this 转换为派生类类型：

```cpp
template <typename Derived>
struct SomeBase
{
    void foo()
    {
        for (auto& item : *static_cast<Derived*>(this))
        {
            ...
        }
    }
}
```

有关此方法的具体示例，请参阅第 9 章。

##### 混合继承

在 C++ 中，类可以定义为继承自它自己的模板参数，例如：

```cpp
template <typename T> struct Mixin : T
{
    ...
}
```

这种方法被称为混合继承（*mixin inheritance*），并允许类型的分层组合。例如，您可以允许 Foo\> x; 声明一个实现所有三个类的特征的类型的变量，而不必实际构造一个全新的 FooBarBaz 类型。

有关此方法的具体示例，请参阅第 9 章。

##### 属性

一个属性（*property*，通常是私有的）仅仅是字段以及 getter 和 setter 的组合。在标准 C++ 中，一个属性如下所示：

```cpp
class Person
{
    int age;
public:
    int get_age() const { return age; }
    void set_age(int value) { age = value; }
};
```

大量的编程语言（例如，C#、Kotlin）通过直接将其添加到编程语言中，将属性的概念内化。虽然 C++ 没有这样做（而且将来也不太可能这样做），但是有一个名为 property 的非标准声明说明符，您可以在大多数编译器（MSVC、Clang、Intel）中使用：

```cpp
class Person
{
    int age_;
public:
    int get_age() const { return age_; }
    void set_age(int value) { age_ = value; }
    __declspec(property(get=get_age, put=set_age)) int age;
};
```

这可以按如下所示使用：

```cpp
Person person;
p.age = 20; // calls p.set_age(20)
```

#### SOLID 设计原则

SOLID 是一个首字母缩写，代表以下设计原则（及其缩写）：

- 单一责任原则（SRP）
- 开闭原则（OCP）
- 里氏替换原则（LSP）
- 接口隔离原则（ISP）
- 依赖注入原则（DIP）

这些原则是由 Robert C. Martin 在 2000 年代初期引入的；事实上，它们只是从 Robert 的书和他的博客中表述的几十项原则中选出的五项原则。这五个特定主题一般都渗透了对模式和软件设计的讨论，所以在我们深入到设计模式之前（我知道你都非常渴望），我们将做一个简短的回顾关于 SOLID 的原则是什么。

##### 单一职责原则

假设您决定把您最私密的想法记在日记里。日记具有一个标题和多个条目。您可以按如下方式对其进行建模：

```cpp
struct Journal
{
    string title;
    vector<string> entries;
    explicit Journal(const string& title) : title{title} {}
};
```

现在，您可以添加用于将添加到日志中的功能，并以日记中的条目序号为前缀。这很容易：

```cpp
void Journal::add(const string& entry)
{
    static int count = 1;
    entries.push_back(boost::lexical_cast<string>(count++)
        + ": " + entry);
}
```

现在，该日记可用于：

```cpp
Journal j{"Dear Diary"};
j.add("I cried today");
j.add("I ate a bug");
```

因为添加一条日记条目是日记实际上需要做的事情，所以将此函数作为 Journal 类的一部分是有意义的。这是日记的责任来保持条目，所以，与这相关的任何事情都是公平的游戏。

现在，假设您决定通过将日记保存在文件中而保留该日记。您需要将此代码添加到 Journal 类：

```cpp
void Journal::save(const string& filename)
{
    ofstream ofs(filename);
    for (auto& s : entries)
        ofs << s << endl;
}
```

这种方法是有问题的。日志的责任是保存日志条目，而不是把它们写道磁盘上。如果您将磁盘写入功能添加到 Journal 和类似类中，持久化方法中的任何更改（例如，您决定向云写入而不是磁盘），都将在每个受影响的类中需要进行大量的微小的更改。

我想在这里停顿一下，并指出：一个架构，使您不得不在大量的类中做很多微小的更改，无论是否相关（如在层次结构中），通常都是一种代码气味（*code smell*）——一个不太对劲的迹象。现在，这完全取决于情况：如果你要重命名一个在 100 个地方使用的符号，我认为这通常是可以的，因为 ReSharper、CLion 或任何你使用的 IDE 实际上将允许你执行重构并且将更改到处传播。但是当你需要完全修改接口时…嗯，那可能是一个非常痛苦的过程！

因此，我指出，持久化是一个单独的问题，最好在一个单独的类别中表达，例如：

```cpp
struct PersistenceManager
{
    static void save(const Journal& j, const string& filename)
    {
        ofstream ofs(filename);
        for (auto& s: j.entries)
            ofs << s << endl;
    }
};
```

这正是单一责任（*Single Responsibility*）的含义：每个类只有一个责任，因此，只有一个改变的理由。只有在需要对条目的存储做更多工作的情况下，Journal 才需要更改。例如，你可能希望每个条目都以时间戳为前缀，因此，你将更改 add() 函数来实现这一点。从另一方面来说，如果你要更改持久化机制，这将在 PersistenceManager 中进行更改。

一个违反 SRP 的反模式的极端例子被称为上帝对象（God Object）。上帝对象是一个巨大的类，它试图处理尽可能多的问题，称为一个难以处理的巨大怪物。

幸运的是，对于我们来说，上帝对象很容易识别出来，并且由于有了源代码管理系统（只需要计算成员函数的数量），负责的开发人员可以迅速确定并受到适当的惩罚。

##### 开闭原则

假设在数据库中，我们拥有一个（完全假设的）范围的产品。每种产品具有颜色和尺寸，并定义为：

```cpp
enum class Color { Red, Green, Blue };
enum class Size { Small, Medium, Large };
struct Product
{
    string name;
    Color color;
    Size size;
};
```

现在，我们希望为给定的一组产品提供特定的过滤功能。我们制作了一个类似于以下内容的过滤器：

```cpp
struct ProductFilter
{
    typedef vector<Product*> Items;
};
```

现在，为了支持通过颜色过滤产品，我们定义了一个成员函数，以精确地执行以下操作：

```cpp
ProductFilter::Items ProductFilter::by_color(Items items, Color color)
{
    Items result;
    for (auto& i : items)
        if (i->color == color)
            result.push_back(i);
    return result;
}
```

我们目前按颜色过滤项目的方法都很好，而且很好。我们的代码开始进入生产环节，但不幸的是，一段时间之后，老板进来并要求我们实现按尺寸大小进行过滤。因此，我们跳回 ProductFilter.cpp 添加以下代码并重新编译：

```cpp
ProductFilter::Items ProductFilter::by_size(Items items, Size size)
{
    Items result;
    for (auto& i : items)
        if (i->size == size)
            result.push_back(i);
    return result;
}
```

这感觉像是彻底的复制，不是吗？为什么我们不直接编写一个接受谓词（一些函数）的通用方法呢？嗯，一个原因可能是不同形式的过滤可以以不同的方式进行：例如，某些记录类型可能被编入索引，需要以特定的方式进行搜索；某些数据类型可以在 GPU 上搜索，而其它数据类型则不适用。

我们的代码进入生成环节，但是，再次的，老板回来告诉我们，现在有一个需求需要按颜色和尺寸进行搜索。那么，我们要做什么呢，还是增加另一个函数？

```cpp
ProductFilter::Items ProductFilter::by_color_and_size(Items items, Size size, Color color)
{
    Items result;
    for (auto& i : items)
        if (i->size == size && i->color == color)
            result.push_back(i);
    return result;
}
```

从前面的场景中，我们想要的是实现“开放-关闭原则”（*Open-Closed Principle*），该原则声明类型是为了扩展而开放的，但为修改而关闭的。换句话说，我们希望过滤是可扩展的（可能在另一个编译单元中），而不必修改它（并且重新编译已经工作并可能已经发送给客户的内容）。

我们如何做到这一点？首先，我们从概念上（SRP!）将我们的过滤过程分为两部分：筛选器（接受所有项并且只返回某些项的过程）和规范（应用于数据元素的谓词的定义）。

我们可以对规范接口做一个非常简单地定义：

```cpp
template <typename T> struct Specification
{
    virtual bool is_satisfied(T* item) = 0;
};
```

在前面的示例中，类型 T 是我们选择的任何类型：它当然可以是一个 Product，但也可以是其它东西。这使得整个方法可重复使用。

接下来，我们需要一种基于 Specification\ 的过滤方法：你猜到的，这是通过定义完成，一个 Filter\：

```cpp
template <typename T> struct Filter
{
    virtual vector<T*> filter(
        vector<T*> items,
        Specification<T>& spec) = 0;
};
```

同样的，我们所做的就是为一个名为 filter 的函数指定签名，该函数接受所有项目和一个规范，并返回符合规范的所有项目。假设这些项目被存储为 vector，但实际上，你可以向 filter() 传递，或者是一对迭代器，或者是一些专门为遍历集合而设计的定制接口。遗憾的是，C++ 语言未能标准化枚举或集合的概念，这是存在于其它编程语言（例如，.NET 的 IEnumerable）中的东西。

基于前述，改进的过滤器的实现非常的简单：

```cpp
struct BetterFilter : Filter<Product>
{
    vector<Product*> filter(
        vector<Product*> items,
        Specification<Product>& spec) override
    {
        vector<Product*> result;
        for (auto& p : items)
            if (spec.is_satisfied(p))
                result.push_back(p);
        return result;
    }
};
```

再次，你可以想到 Specification\，该规范被传入作为 std::function 的强类型化等效项，该函数仅约束到一定数量的可能的筛选规格。

现在，这是最简单的部分。为了制作一个颜色过滤器，你可以制作一个 ColorSpecification：

```cpp
struct ColorSpecification : Specification<Product>
{
    Color color;
    explicit ColorSpecification(const Color color) :
    color{color} {}
    bool is_satisfied(Product* item) override {
        return item->color == color;
    }
};
```

根据本规范，以及给定的产品清单，我们现在可以按如下方式过滤这些产品：

```cpp
Product apple{ "Apple", Color::Green, Size::Small };
Product tree{ "Tree", Color::Green, Size::Large };
Product house{ "House", Color::Blue, Size::Large };
vector<Product*> all{ &apple, &tree, &house };
BetterFilter bf;
ColorSpecification green(Color::Green);
auto green_things = bf.filter(all, green);
for (auto& x : green_things)
    cout << x->name << " is green" << endl;
```

前面给我们的是 “Apple” 和 “Tree”，因为它们都是绿色的。现在，我们迄今为止尚未实现的唯一目标是搜索尺寸和颜色（或者，实际上，解释了如何搜索尺寸或颜色，或混合不同的标准）。答案是你简单地做了一个复合规范。例如，对于逻辑 AND，你可以使其如下所示：

```cpp
template <typename T> struct AndSpecification :
Specification<T>
{
    Specification<T>& first;
    Specification<T>& second;
    AndSpecification(Specification<T>& first,
                     Specification<T>& second)
        : first{first}, second{second} {}
    bool is_satisfied(T* item) override
    {
        return first.is_satisfied(item) && second.is_satisfied(item);
    }
};
```

现在，你可以在更简单的规范基础上创建复合条件。复用我们早期制作的绿色规范，找到一些绿色和大的东西现在就像这样简单：

```cpp
SizeSpecification large(Size::Large);
ColorSpecification green(Color::Green);
AndSpecification<Product> green_and_large{ large, green };
auto big_green_things = bf.filter(all, green_and_big);
for (auto& x : big_green_things)
    cout << x->name << " is large and green" << endl;
// Tree is large and green
```

这里有很多代码！但是请记住，由于 C++ 的强大功能，你可以简单地引入一个 operator && 用于两个 Specification\ 对象，从而使得过滤过程由两个（或更多！）标准，极为简单：

```cpp
template <typename T> struct Specification
{
    virtual bool is_satisfied(T* item) = 0;
    AndSpecification<T> operator &&(Specification&& other)
    {
        return AndSpecification<T>(*this, other);
    }
};
```

如果你现在避免为尺寸/颜色规范设置额外的变量，则可以将复合规范简化为一行：

```cpp
auto green_and_big =
    ColorSpecification(Color::Green)
    && SizeSpecification(Size::Large);
```

因此，让我们回顾以下 OCP 原则是声明，以及前面的示例是如何执行它的。基本上，OCP 声明你不需要返回你已经编写和测试过的代码，并对其进行更改。这正是这里发生的！我们制定了 Specification\ 和 Filter\，从那时起，我们所要做的就是实现任何一个接口（不需要修改接口本身）来实现新的过滤机制。这就是“开放供扩展，封闭供修改”的意思。

##### 里氏替换原则

里氏替换原则（以 Barbara Liskov 命名）指出，如果一个接口可以接受类型为 Parent 的对象，那么它应该同样地可以接受类型为 Child 的对象，而不会有任何破坏。让我们来看看 LSP 被破坏的情况。

下面是一个矩形；它有宽度（width）和高度（height），以及一组计算面积的 getters 和 setters：

```cpp
class Rectangle
{
protected:
    int width, height;
public:
    Rectangle(const int width, const int height)
        : width{width}, height{height} { }
    int get_width() const { return width; }
    virtual void set_width(const int width) { this->width = width; }
    int get_height() const { return height; }
    virtual void set_height(const int height) { this->height = height; }
    int area() const { return width * height; }
};
```

现在，假设我们有一种特殊的矩形，称为正方形。此对象将重写 setters，以设置宽度和高度：

```cpp
class Square : public Rectangle
{
public:
    Square(int size) : Rectangle(size, size) {}
    void set_width(const int width) override {
        this->width = height = width;
    }
    void set_height(const int height) override {
        this->height = width = height;
    }
};
```

这种做法是邪恶的。你还看不到它，因为它确实是无辜的：setters 简单地设置了两个维度，可能会发生什么错误呢？好吧，如果我们采用前面的方法，我们可以很容易地构建一个函数，该函数以 Rectangle 类型变量为参数，当传入 Square 类型变量时，它会爆炸：

```cpp
void process(Rectangle& r)
{
    int w = r.get_width();
    r.set_height(10);
    cout << "expected area = " << (w * 10)
        << ", got " << r.area() << endl;
}
```

前面的函数以公式 Area = Width * Height 作为不变量。它得到宽度，设置高度，并正确地期望乘积等于计算的面积。但是使用 Square 调用前面的函数会产生不匹配：

```cpp
Square s{5};
process(s); // expected area = 50, got 25
```

从这个例子（我承认有点人为的）得到的启示是，process() 完全不能接受派生类型 Square 而不是基类型 Rectangle，从而破坏了 LSP 原则。如果你给它一个 Rectangle，一切都很好，所以它可能需要一些时间才能出现在你的测试（或者生产，希望不是！）。

解决办法是什么呢？嗯，有很多。就我个人而言，我认为类型 Square 甚至不应该存在：相反，我们可以创建一个工厂（参见第3章）来创建矩形和正方形：

```cpp
struct RectangleFactory
{
    static Rectangle create_rectangle(int w, int h);
    static Rectangle create_square(int size);
};
```

你也可能需要一种检测一个 Rectangle 是否是一个 Square 的方法：

```cpp
bool Rectangle::is_square() const
{
    return width == height;
}
```

在这种情况下，核心选项是在 Square 的 set_width() / set_height() 中抛出一个异常，说明这些操作不受支持，你应该使用 set_size() 代替。但是，这违反了最小覆盖的原则（ *principle of least surpise*），因为你希望调用 set_width() 来进行有意义的更改…我说的对吗？

##### 接口分离原则

好吧，这是另一个人为的例子，尽管如此，它仍然适合于说明这个问题。假设你决定定义一个多功能打印机：该设备可以打印、扫描和传真文档。因此，你可以定义如下：

```cpp
struct MyFavouritePrinter /* : IMachine */
{
    void print(vector<Document*> docs) override;
    void fax(vector<Document*> docs) override;
    void scan(vector<Document*> docs) override;
};
```

这很好。现在，假设你决定定义一个需要由所有计划制造多功能打印机的人实现的接口。因此，你可以在你最喜欢的 IDE 中使用提取接口函数功能，你可以得到如下内容：

```cpp
struct IMachine
{
    virtual void print(vector<Document*> docs) = 0;
    virtual void fax(vector<Document*> docs) = 0;
    virtual void scan(vector<Document*> docs) = 0;
};
```

这里有一个问题。原因是这个接口的一些实现者可能不需要扫描或传真，只需要打印。然而，你强迫他们实现这些额外的功能：当然，它们可以都是无操作的，但为什么要这么做呢？

因此，ISP 的建议是将接口分开，以便于实现者可以根据他们的需求进行挑选和选择。由于打印和扫描是不同的操作（例如，扫描仪不能打印），我们为这些操作定义了不同的接口：

```cpp
struct IPrinter
{
    virtual void print(vector<Document*> docs) = 0;
};
struct IScanner
{
    virtual void scan(vector<Document*> docs) = 0;
};
```

然后，打印机或扫描仪可以实现所需的功能：

```cpp
struct Printer : IPrinter
{
    void print(vector<Document*> docs) override;
};
struct Scanner : IScanner
{
    void scan(vector<Document*> docs) override;
};
```

现在，如果我们真的想要一个 IMachine 接口，我们可以将它定义为上述接口的组合：

```cpp
struct IMachine : IPrinter, IScanner /* IFax and so on */
{  
};
```

当你在具体的多功能设备中实现这个接口时，这就是要使用的接口。例如，你可以使用简单的委托来确保 Machine 重用特定 IPrinter 和 IScanner 提供的功能：

```cpp
struct Machine : IMachine
{
    IPrinter& printer;
    IScanner& scanner;
    Machine(IPrinter& printer, IScanner& scanner)
        : printer{printer},
          scanner{scanner}
    {
    }
    void print(vector<Document*> docs) override
    {
        printer.print(docs);
    }
    void scan(vector<Document*> docs) override
    {
        scanner.scan(docs);
    }
};
```

因此，简单地说，这里的想法是将复杂接口的部分分隔成单独的接口，以避免迫使实现者实现他们并不真正需要的功能。当为某些复杂的应用程序编写插件时，如果你得到一个具有 20 个令人困惑的函数的接口，用于实现各种 no-ops 和 return nullptr 时，说不定是 API 作者违反了 ISP 原则。

##### 依赖反转原则

DIP 的原始定义如下所示[^3] ：

*A. High-level modules should not depend on low-level modules. Both should depend on abstractions*.

注3：Martin, Robert C., *Agile Software Development, Principles, Patterns, and Practices* (New York: Prentice Hall, 2003), pp. 127-131.

这句话的主要意思是，如果你对日志记录感兴趣，你的报告组件不应该依赖于具体的 ConsoleLogger，而是可以依赖于 ILogger 接口。在这种情况下，我们认为报告组件是高级别的（更接近业务领域），而日志记录则是一个基本的关注点（类似于文件 I/O 或线程，但不是），被认为是一个低级别的模块。

*B. Abstractions should not depend on details. Details should depend on abstractions.*

这再次重申了接口或基类上的依赖比依赖于具体的类型更好。希望这个语句的真实性是显而易见的，因为这种方法支持更好的可配置性和可测试性——前提是你使用了一个良好的框架来处理这些依赖关系。

所以，现在的主要问题是：你是如何真正实现上述所有的？这确实需要更多的工作，因为现在你需要明确说明，例如，Reporting 依赖于 ILogger。你表达它的方式也许如下所示：

```cpp
class Reporting
{
    ILogger& logger;
public:
    Reporting(const ILogger& logger) : logger{logger} {}
    void prepare_report()
    {
        logger.log_info("Preparing the report");
        ...
    }
};
```

现在的问题是，要初始化前面的类，你需要显式地调用 Reporting{ConsoleLogger{}} 或类似地东西。如果 Reporting 依赖于五个不同的接口呢？如果 ConsoleLogger 有自己的依赖项，又怎么办？你可以通过编写大量的代码来管理这个问题，但是这里有一个更好的方法。

针对前面的现代、流行、时尚的做法是使用依赖注入（*Dependency Injection*）：这基本上意味着你要使用诸如 Boost.DI[^4]之类的库自动满足特定组件的依赖关系的要求。

注4：At the moment, Boost.DI is not yet part of Boost proper, it is part of the boost-experimental Github repository.

让我们考虑一个具有引擎但还需要写入日志的汽车的例子。从目前的情况来看，我们可以说一辆车取决于这两件情况。首先，我们可以将引擎定义为：

```cpp
struct Engine
{
    float volume = 5;
    int horse_power = 400;
    friend ostream& operator<< (ostream& os, const Engine& obj)
    {
        return os
            << "volume: " << obj.volume
            << "horse_power: " << obj.horse_power;
    } // thanks, ReSharper!
};
```

现在，由我们决定是否要提取一个 IEngine 接口并将其馈送到汽车。也许我们有，也许我们没有，这通常是一个设计决定。如果你设想有一个引擎层次结构，或者你预见到为了测试目的需要一个 NullEngine（参见第19章），那么是的，你确实需要抽象出接口。

无论如何，我们也需要日志记录，因为这可以通过多种方式完成（控制台、电子邮件、短信、鸽子邮件…），我们可能希望有一个 ILogger 接口：

```cpp
struct ILogger
{
    virtual ~ILogger() {}
    virtual void Log(const string& s) = 0;
};
```

以及某种具体的实现：

```cpp
struct ConsoleLogger : ILogger
{
    ConsoleLogger() {}
    void Log(const string& s) override
    {
        cout << "LOG: " << s.c_str() << endl;
    }
};
```

现在，我们将要定义的汽车取决于引擎和日志组件。我们两者都需要，但这取决于我们如何存储它们：我们可以使用指针，引用，unique_ptr/shared_ptr 或其它。我们将这两个依赖组件定义为构造函数的参数：

```cpp
struct Car
{
    unique_ptr<Engine> engine;
    shared_ptr<ILogger> logger;
    Car(unique_ptr<Engine> engine,
        const shared_ptr<ILogger>& logger)
      : engine{move(engine)},
        logger{logger}
    {
        logger->Log("making a car");
    }
    friend ostream& operator<<(ostream& os, const Car& obj)
    {
        return os << "car with engine: " << *obj.engine;
    }
};
```

现在，你可能希望在初始化 Car 时看到对 make_unique/make_shared 的调用。但我们不会这么做的。相反，我们将使用 Boost.DI。首先，我们将定义一个绑定，将 ILogger 绑定到 ConsoleLogger；这意味着，只要有人要求一个 ILogger，就给他们一个 ConsoleLogger：

```cpp
auto injector = di::make_injector(
    di::bind<ILogger>().to<ConsoleLogger>()
);
```

现在，我们已经配置了注射器，我们可以使用它来创建一辆汽车：

```cpp
auto car = injector.create<shared_ptr<Car>>();
```

前面的内容创建了一个 shared_ptr\，它指向了一个完全初始化的 Car 对象，这正是我们想要的。这种方法的伟大之处在于，如果需要更改正在使用的记录器的类型，我们可以在一个地方（绑定调用）更改它，而 ILogger 出现的每个地方现在都可以使用我们提供的其它日志记录组件。这种方法还可以帮助我们进行单元测试，并允许我们使用桩（或 Null 对象模式）代替模拟。

##### 模式时间!

通过对 SOLID 设计原则的理解，我们将深入到设计模式本身。请系好安全带，这将是一段漫长的旅程（希望不会很无聊）。



## 第二章 建造者模式

### 建造者模式

建造者模式（`Builder`）涉及到复杂对象的创建，即不能在单行构造函数调用中构建的对象。这些类型的对象本身可能由其他对象组成，可能涉及不太明显的逻辑，需要一个专门用于对象构造的单独组件。

我认为值得事先注意的是，虽然我说建造者适用于复杂的对象的创建，但我们将看一个相当简单的示例。这样做纯粹是为了空间优化，这样领域逻辑的复杂性就不会影响读者欣赏模式实现的能力。

#### 场景

让我们想象一下，我们正在构建一个呈现`web`页面的组件。首先，我们将输出一个简单的无序列表，其中有两个`item`，其中包含单词`hello`和`world`。一个非常简单的实现如下所示:

```cpp
string words[] = { "hello", "world" };
ostringstream oss;
oss << "<ul>";
for (auto w : words)
oss << " <li>" << w << "</li>";
oss << "</ul>";
printf(oss.str().c_str())
```

这实际上给了我们想要的东西，但是这种方法不是很灵活。如何将项目符号列表改为编号列表?在创建了列表之后，我们如何添加另一个`item`?显然，在我们这个死板的计划中，这是不可能的。

因此，我们可以通过`OOP`的方法定义一个`HtmlElement`类来存储关于每个`tag`的信息:

```cpp
struct HtmlElement {
  string name;
  string text;
  vector<HtmlElement> elements;
  HtmlElement() {}
  HtmlElement(const string& name, const string& text)
      : name(name), text(text) {}
  string str(int indent = 0) const {
    // pretty-print the contents
  }
};
```

有了这种方法，我们现在可以以一种更合理的方式创建我们的列表:

```cpp
string words[] = {"hello", "world"};
HtmlElement list{"ul", ""};
for (auto w : words) list.elements.emplace_back{HtmlElement{"li", w}};
printf(list.str().c_str());
```

这做得很好，并为我们提供了一个更可控的、`OOP`驱动的条目列表表示。但是构建每个`HtmlElement`的过程不是很方便，我们可以通过实现建造者模式来改进它。

#### 简单建造者

建造者模式只是试图将对象的分段构造放到一个单独的类中。我们的第一次尝试可能会产生这样的结果：

```cpp
struct HtmlBuilder {
  HtmlElement root;
  HtmlBuilder(string root_name) { root.name = root_name; }
  void add_child(string child_name, string child_text) {
    HtmlElement e{child_name, child_text};
    root.elements.emplace_back(e);
  }
  string str() { return root.str(); }
};
```

这是一个用于构建`HTML`元素的专用组件。`add_child()`方法是用来向当前元素添加额外的子元素的方法，每个子元素都是一个名称-文本对。它可以如下使用:

```cpp
HtmlBuilder builder{ "ul" };
builder.add_child("li", "hello");
builder.add_child("li", "world");
cout << builder.str() << endl;
```

你会注意到，此时`add_child()`函数是返回空值的。我们可以使用返回值做许多事情，但返回值最常见的用途之一是帮助我们构建流畅的接口。

#### 流畅的建造者

让我们将`add_child()`的定义改为如下:

```cpp
HtmlBuilder& add_child(string child_name, string child_text) {
  HtmlElement e{child_name, child_text};
  root.elements.emplace_back(e);
  return *this;
}
```

通过返回对建造者本身的引用，现在可以在建造者进行链式调用。这就是所谓的流畅接口(`fluent interface`):

```cpp
HtmlBuilder builder{ "ul" };
builder.add_child("li", "hello").add_child("li", "world");
cout << builder.str() << endl;
```

引用或指针的选择完全取决于你。如果你想用`->`操作符，可以像这样定义`add_child()` 

```cpp
HtmlBuilder* add_child(string child_name, string child_text) {
  HtmlElement e{child_name, child_text};
  root.elements.emplace_back(e);
  return this;
}
```

像这样使用：

```cpp
HtmlBuilder builder = new HtmlBuilder{ "ul" };
builder->add_child("li", "hello")->add_child("li", "world");
cout << builder->str() << endl;
```

#### 交流意图

我们为`HTML`元素实现了一个专用的建造者，但是我们类的用户如何知道如何使用它呢?一种想法是，只要他们构造对象，就强制他们使用建造者。你需要这样做:

```cpp
struct HtmlElement {
  string name;
  string text;
  vector<HtmlElement> elements;
  const size_t indent_size = 2;
  static unique_ptr<HtmlBuilder> build(const string& root_name) {
    return make_unique<HtmlBuilder>(root_name);
  }
 protected:  // hide all constructors
  HtmlElement() {}
  HtmlElement(const string& name, const string& text)
      : name{name}, text{text} {}
};
```

我们的做法是双管齐下。首先，我们隐藏了所有的构造函数，因此它们不再可用。但是，我们已经创建了一个工厂方法(这是我们将在后面讨论的设计模式)，用于直接从`HtmlElement`创建一个建造者。它也是一个静态方法。下面是如何使用它：

```cpp
auto builder = HtmlElement::build("ul"); 
(*builder).add_child("li", "hello").add_child("li", "world");
cout << builder.str() << endl;
```

但是不要忘记，我们的最终目标是构建一个`HtmlElement`，而不仅仅是它的建造者!因此，锦上添花的可能是建造者上的 `operator HtmlElement`的实现，以产生最终值:

```cpp
struct HtmlBuilder {
  operator HtmlElement() const { return root; }
  HtmlElement root;
  // other operations omitted
};
```

前面的一个变体是返回std::move(root)，但是否这样做实际上取决于你自己。不管怎样，运算符的添加允许我们写下以下内容:

```cpp
HtmlElement e = *(HtmlElement::build("ul"))
                     .add_child("li", "hello")
                     .add_child("li", "world");
cout << e.str() << endl;
```







