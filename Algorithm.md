# 算法



## 十大经典排序算法

[1.0 十大经典排序算法 | 菜鸟教程 (runoob.com)](https://www.runoob.com/w3cnote/ten-sorting-algorithm.html)



### 冒泡排序（Bubble Sort）

冒泡排序是一种简单的排序算法。它重复地走访过要排序的数列，一次比较两个元素，如果它们的顺序错误就把它们交换过来。走访数列的工作是重复地进行指导没有再需要交换，也就是说该数列已经排序完成。这个算法的名字由来是因为越小的元素会经由交换慢慢“浮”到数列的顶端。

#### 算法描述

- 比较相邻的元素，如果第一个比第二个大，就交换它们两个；
- 对每一对相邻的元素做同样的工作，从开始第一队到结尾的最后一对，这样在最后的元素应该回事最大的数；
- 针对所有元素重复以上步骤，出来最后一个；
- 持续每次对越来越少的元素重复上面的步骤，知道没有任何一对数字需要比较。

#### 动图演示



![849589-20171015223238449-2146169197](Algorithm.assets/849589-20171015223238449-2146169197.gif)



#### 什么时候最快

当输入的数据已经是正序时（都已经时正序了，我还要你冒泡排序有何作用啊）。

#### 什么时候最慢

当输入的数据是反序时（写一个 for 循环反序输出数据不就行了，干嘛要用你冒泡排序呢，我是闲的吗）。

####  代码实现

**C语言**

```c
#include <stdio.h>
void bubble_sort(int arr[], int len) {
        int i, j, temp;
        for (i = 0; i < len - 1; i++)
                for (j = 0; j < len - 1 - i; j++)
                        if (arr[j] > arr[j + 1]) {
                                temp = arr[j];
                                arr[j] = arr[j + 1];
                                arr[j + 1] = temp;
                        }
}
int main() {
        int arr[] = { 22, 34, 3, 32, 82, 55, 89, 50, 37, 5, 64, 35, 9, 70 };
        int len = (int) sizeof(arr) / sizeof(*arr);
        bubble_sort(arr, len);
        int i;
        for (i = 0; i < len; i++)
                printf("%d ", arr[i]);
        return 0;
}
```

**C++**

```cpp
#include <iostream>
using namespace std;
template<typename T> //整数或浮点数皆可使用,若要使用类(class)或结构体(struct)时必须重载大于(>)运算符
void bubble_sort(T arr[], int len) {
        int i, j;
        for (i = 0; i < len - 1; i++)
                for (j = 0; j < len - 1 - i; j++)
                        if (arr[j] > arr[j + 1])
                                swap(arr[j], arr[j + 1]);
}
int main() {
        int arr[] = { 61, 17, 29, 22, 34, 60, 72, 21, 50, 1, 62 };
        int len = (int) sizeof(arr) / sizeof(*arr);
        bubble_sort(arr, len);
        for (int i = 0; i < len; i++)
                cout << arr[i] << ' ';
        cout << endl;
        float arrf[] = { 17.5, 19.1, 0.6, 1.9, 10.5, 12.4, 3.8, 19.7, 1.5, 25.4, 28.6, 4.4, 23.8, 5.4 };
        len = (float) sizeof(arrf) / sizeof(*arrf);
        bubble_sort(arrf, len);
        for (int i = 0; i < len; i++)
                cout << arrf[i] << ' '<<endl;
        return 0;
}
```

**python**

```python
def bubbleSort(arr):
    for i in range(1, len(arr)):
        for j in range(0, len(arr)-i):
            if arr[j] > arr[j+1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
```





### 选择排序

选择排序是一种简单直观的排序算法，无论什么数据进去都是O($n^2$)的时间复杂度。所以用到它的时候，数据规模越小越好。唯一的好处可能就是不占用额外的内存空间了吧。

#### 算法步骤

- 首先在未排序序列种找到最小（大）元素，存放到排序序列的起始位置；
- 再从剩余未排序元素中继续寻找最小（大）元素，然后放到已排序列的末尾；
- 重复第二步，直到所有元素均排序完毕。

#### 动图演示

![selectionSort](Algorithm.assets/selectionSort.gif)



#### 代码实现

**python代码实现**

```python
def selectionSort(arr):
    for i in range(len(arr) - 1):
        # 记录最小数的索引
        minIndex = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[minIndex]:
                minIndex = j
        # i 不是最小数时，将 i 和最小数进行交换
        if i != minIndex:
            arr[i], arr[minIndex] = arr[minIndex], arr[i]
    return arr
```

**C 语言**

```c
void swap(int *a,int *b) //交换两个变数
{
    int temp = *a;
    *a = *b;
    *b = temp;
}
void selection_sort(int arr[], int len)
{
    int i,j;

        for (i = 0 ; i < len - 1 ; i++)
    {
                int min = i;
                for (j = i + 1; j < len; j++)     // 走访未排序的元素
                        if (arr[j] < arr[min])    //找到目前最小值
                                min = j;    //记录最小值
                swap(&arr[min], &arr[i]);    //做交换
        }
}
```

**C++**

```cpp
template <typename T> // 整数或者浮点数皆可使用，若要使用类（class)时，必须设定大于(>)的运算子功能
void select_sort(std::vector<T> & arr)
{
    for(int i = 0; i < arr.size() - 1; i++)
    {
        int min = i;
        for(int j = i + 1; j < arr.size(); j++)
            if(arr[j] < arr[min])
                min = j;
        std::swap(arr[i], arr[min]);
    }
}
```

### 插入排序

插入排序的代码实现虽然没有冒泡排序和选择排序那么简单粗暴，但它的原理应该是最容易理解的。因为只要打过扑克牌的人都应该能够秒懂。插入排序是一种简单直观的排序算法，它的工作原理是通过构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。

插入排序和冒泡排序一样，也有一种优化算法，叫做拆半插入。

#### 算法步骤

将第一待排序序列第一个元素看作一个有序序列，把第二个元素到最后一个元素当成是未排序序列。

从头到尾依次扫描未排序序列，将扫描到的每个元素插入有序序列的适当位置。（如果待插入的元素与有序序列中的某个元素相等，则将待插入元素插入到相等元素的后面。）



#### 动图演示



![insertionSort](Algorithm.assets/insertionSort.gif)



#### 代码实现

python

```python
def insertionSort(arr):
    for i in range(len(arr)):
        preIndex = i-1
        current = arr[i]
        while preIndex >= 0 and arr[preIndex] > current:
            arr[preIndex+1] = arr[preIndex]
            preIndex-=1
        arr[preIndex+1] = current
    return arr
```

C

```c
void insertion_sort(int arr[], int len){
        int i,j,key;
        for (i=1;i<len;i++){
                key = arr[i];
                j=i-1;
                while((j>=0) && (arr[j]>key)) {
                        arr[j+1] = arr[j];
                        j--;
                }
                arr[j+1] = key;
        }
}
```

C++

```cpp
void insertion_sort(int arr[],int len){
        for(int i=1;i<len;i++){
                int key=arr[i];
                int j=i-1;
                while((j>=0) && (key<arr[j])){
                        arr[j+1]=arr[j];
                        j--;
                }
                arr[j+1]=key;
        }
}
```

































































