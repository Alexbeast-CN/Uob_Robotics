#! https://zhuanlan.zhihu.com/p/422599653
# Week1. Machine Vision Tutorial - Python 

> 本篇笔记的内容基于[斯坦福的 Colab Notes](https://colab.research.google.com/github/cs231n/cs231n.github.io/blob/master/python-colab.ipynb#scrollTo=dzNng6vCL9eP)

> colab的本质是一个 web 版的 python ide + Markdown 编辑器。而从我的个人习惯来说，我更喜欢使用 Vscode 作为我的编辑器，但可惜的是我的电脑（ThinkPad X1C）没有强大的 GPU 所以本门课程最后关于卷积神经网络的部分依然会在 Colab 中完成。

>文中所提及的 Colab 在中国无法'正常'使用

## Python Tutorial With Google Colab

> Documentation of this tutorial:
> - [python build-in type](https://docs.python.org/3.7/library/stdtypes.html#numeric-types-int-float-long-complex)
> - [A list of all string methods](https://docs.python.org/3.7/library/stdtypes.html#string-methods)
> - [lists](https://docs.python.org/3.7/tutorial/datastructures.html#more-on-lists)
> - [dictionaries](https://docs.python.org/2/library/stdtypes.html#dict)
> - [numpy data type](https://numpy.org/doc/stable/reference/arrays.dtypes.html)
> - [mathematical functions of numpy](https://numpy.org/doc/stable/reference/routines.math.html)
> - [broadcasting](https://numpy.org/doc/stable/reference/ufuncs.html#available-ufuncs)

### Basics of Python

Python 是一种高级的、动态类型的多范式编程语言。Python 代码通常被认为几乎像伪代码，因为它允许您用很少的代码行表达非常强大的想法，同时又非常易读。例如，这是在 Python 中经典快速排序算法的实现：

```py
import numpy
import cv2

print(cv2.__version__)
print(numpy.__version__)

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print(quicksort([3,6,8,10,1,2,1]))  
```


```
<string>:1: UserWarning: The NumPy module was reloaded (imported a second time). This can, in some cases, result in small but subtle issues and is discouraged.
4.5.3
1.21.2
[1, 1, 2, 3, 6, 8, 10]
```

### Containers

#### List

A list is the Python equivalent of an array but is resizeable and can contain elements of different types:

```py
xs = [3, 1, 2]   # Create a list
print(xs, xs[2])
# -1 means the end of the list
print(xs[-1])     # Negative indices count from the end of the list; prints "2."
```



```
[3, 1, 2] 2
2
```

```py
xs[2] = 'foo'    # Lists can contain elements of different types
print(xs)
```


```
[3, 1, 'foo']
```

```py
xs.append('bar') # Add a new element to the end of the list
print(xs)
```


```
[3, 1, 'foo', 'bar']
```

```py
x = xs.pop()     # Remove and return the last element of the list
print(x, xs)
```


```
bar [3, 1, 'foo']
```

For more reference, check [Python Documents](https://docs.python.org/3.7/)

#### Loops

You can loop over the elements of a list like this:

```py
animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)
```


```
cat
dog
monkey
```

If you want access to the index of each element within the body of a loop, use the built-in enumerate function:

```py
animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print('#{}: {}'.format(idx + 1, animal))
```


```
#1: cat
#2: dog
#3: monkey
```

#### List Comprehension:

在编程时，我们经常希望将一种类型的数据转换为另一种类型的数据。作为一个简单的例子，考虑以下计算平方数的代码：

```py
# Block 1
nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print(squares)
```

```py
# Block 2
nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print(squares)
```

`Block 1` 与 `Block 2` 的写法等效，输出结果同为：


```
[0, 1, 4, 9, 16]
```

List Comprehension 里也可以添加条件：

```py
nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(even_squares)
```

#### Dictionaries

字典存储（键，值）对，类似于 Java 中的 Map 或 Javascript 中的对象。你可以这样使用它：

```py
d = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
print(d['cat'])       # Get an entry from a dictionary; prints "cute"
print('cat' in d)     # Check if a dictionary has a given key; prints "True"
```


```
cute
True
```

```py
d['fish'] = 'wet'    # Set an entry in a dictionary
print(d['fish'])      # Prints "wet"
```


```
wet
```

如果我们用字典中没有的键，则会报错。但是，我们可以在打印的时候给其一个默认值，则会将默认值打印出来。比如：

```py
print(d.get('monkey', 'N/A'))  # Get an element with a default; prints "N/A"
print(d.get('fish', 'N/A'))    # Get an element with a default; prints "wet"
```

:
```
N/A
wet
```

删除字典中的键

```py
del d['fish']        # Remove an element from a dictionary
print(d.get('fish', 'N/A')) # "fish" is no longer a key; prints "N/A"
```

:
```
N/A
```

遍历字典中的键很容易：

>items()
>Return a copy of the dictionary’s list of (key, value) pairs.

```py
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal, legs in d.items():
    print('A {} has {} legs'.format(animal, legs))
```

:
```
A person has 2 legs
A cat has 4 legs
A spider has 8 legs
```

#### Sets

集合是不同元素的无序集合。作为一个简单的例子，请考虑以下内容：

```py
animals = {'cat', 'dog'}
print('cat' in animals)   # Check if an element is in a set; prints "True"
print('fish' in animals)  # prints "False"
```

:
```
True
False
```

```py
animals.add('fish')      # Add an element to a set
print('fish' in animals)
print(len(animals))       # Number of elements in a set;
```

:
```
True
3
```

```py
animals.add('cat')       # Adding an element that is already in the set does nothing
print(len(animals))       
animals.remove('cat')    # Remove an element from a set
print(len(animals))
```

> 已经存在于集合中的元素不可被重复添加

:
```
3
2
```

循环：迭代集合与迭代列表的语法相同；然而，由于集合是无序的，您不能对访问集合元素的顺序做出假设：

```py
animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print('#{}: {}'.format(idx + 1, animal))x
```


:

```
#1: cat
#2: fish
#3: dog
```

#### Tuples

元组是一个（不可变的）有序值列表。元组在很多方面类似于列表；最重要的区别之一是元组可以用作字典中的键和集合的元素，而列表则不能。这是一个简单的例子：

```py
# 冒号左边是键，右边是值
d = {(x, x + 1): x for x in range(10)}  # Create a dictionary with tuple keys
t = (5, 6)       # Create a tuple
print(type(t))
print(d[t])       
print(d[(1, 2)])
```

Variables:
```
    d: -{
        (0, 1): 0,
        (1, 2): 1,
        (2, 3): 2,
        (3, 4): 3,
        (4, 5): 4,
        (5, 6): 5,
        (6, 7): 6,
        (7, 8): 7,
        (8, 9): 8,
        (9, 10): 9
```

:
```
<class 'tuple'>
5
1
```

### Functions

Python 函数是使用def关键字定义的。例如：

```py
def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]:
    print(sign(x))
```

:
```
negative
zero
positive
```

我们经常会定义函数来接受可选的关键字参数，如下所示：

```py
def hello(name, loud=False):
    if loud:
        print('HELLO, {}'.format(name.upper()))
    else:
        print('Hello, {}!'.format(name))

hello('Bob')
hello('Fred', loud=True)
```

:
```
Hello, Bob!
HELLO, FRED
```

### Classes:

在 Python 中定义类的语法很简单：

```py
class Greeter:

    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable

    # Instance method
    def greet(self, loud=False):
        if loud:
          print('HELLO, {}'.format(self.name.upper()))
        else:
          print('Hello, {}!'.format(self.name))

g = Greeter('Fred')  # Construct an instance of the Greeter class
g.greet()            # Call an instance method; prints "Hello, Fred"
g.greet(loud=True)   # Call an instance method; prints "HELLO, FRED!"
```

```
Hello, Fred!
HELLO, FRED
```

### Numpy

Numpy 是 Python 科学计算的核心库。它提供了一个高性能的多维数组对象，以及用于处理这些数组的工具。


使用 Numpy 创建数组：

```py
a = np.array([1, 2, 3])  # Create a rank 1 array
print(type(a), a.shape, a[0], a[1], a[2])
a[0] = 5                 # Change an element of the array
print(a)              
```

:
```
<class 'numpy.ndarray'> (3,) 1 2 3
[5 2 3]
```

```py
b = np.array([[1,2,3],[4,5,6]])   # Create a rank 2 array
print(b)
print(" ")
print(b.shape)
print(b[0, 0], b[0, 1], b[1, 0])
```

:
```
[[1 2 3]
 [4 5 6]]

 (2, 3)
1 2 4
```

Numpy also provides many functions to create arrays:

```py
a =np.zeros((3,3))  # Create an array of all zeros
b = np.ones((3,3)) # Create an array of all ones
c = np.eye(2)  # Create a 2x2 identity matrix
d = np.full((3,3),9) # Create a constant array
e = np.random.random((2,2)) # Create an array filled with random values
print(a,"\n\n", b,"\n\n", c,"\n\n", d,"\n\n", e)
```

#### Array indexing

Numpy 提供了多种索引数组的方法。

切片：与 Python 列表类似，可以对 numpy 数组进行切片。由于数组可能是多维的，您必须为数组的每个维度指定一个切片：

```py
import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
b = a[:2, 1:3]
print(b)
```

> 注意：这里 [1:3] 只取了第1，2列

:
```
[[2 3]
 [6 7]]
```

数组的切片是相同数据的视图，因此修改它会修改原始数组。

您还可以将整数索引与切片索引混合使用。但是，这样做会产生一个比原始数组等级低的数组。请注意，这与 MATLAB 处理数组切片的方式大不相同：

访问数组中间行数据的两种方式。将整数索引与切片混合会产生一个较低等级的数组，而仅使用切片会产生与原始数组具有相同等级的数组：

```py
# Create the following rank 2 array with shape (3, 4)
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a)

row_r1 = a[1, :]    # Rank 1 view of the second row of a  
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
row_r3 = a[[1], :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)
print(row_r2, row_r2.shape)
print(row_r3, row_r3.shape)
```

:

```
[5 6 7 8] (4,)
[[5 6 7 8]] (1, 4)
[[5 6 7 8]] (1, 4)
```

我们也可以使用同样的方法对列进行操作。

整数数组索引：当您使用切片索引到 numpy 数组时，生成的数组视图将始终是原始数组的子数组。相比之下，整数数组索引允许您使用来自另一个数组的数据构造任意数组。下面是一个例子：

```py
a = np.array([[1,2], [3, 4], [5, 6]])

# An example of integer array indexing.
# The returned array will have shape (3,) and 
#      a[[行],[列]]
print(a[[1,0,0],[1,0,0]])

# The above example of integer array indexing is equivalent to this:
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))
```

：

```
[4 1 1]
[1 4 5]
```

也可以使用数组来索引数：

```py
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])

# Create an array of indices
b = np.array([0, 2, 0, 1])

# Select one element from each row of a using the indices in b
print(a[np.arange(4), b])  # Prints "[ 1  6  7 11]"
```

:

```
[ 1  6  7 11]
```

#### Array math

基本数学函数在数组上按元素操作，可用作运算符重载和 numpy 模块中的函数：

元素加法：

```py
x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise sum; both produce the array
print(x + y)
print(np.add(x, y))
```

:
```
[[ 6.  8.]
 [10. 12.]]
[[ 6.  8.]
 [10. 12.]]
 ```

元素减法：

```py
# Elementwise difference; both produce the array
print(x - y)
print(np.subtract(x, y))
```

:
```
[[-4. -4.]
 [-4. -4.]]
[[-4. -4.]
 [-4. -4.]]
```

元素乘法：

```py
# Elementwise product; both produce the array
print(x * y)
print(np.multiply(x, y))
```

:
```
[[ 5. 12.]
 [21. 32.]]
[[ 5. 12.]
 [21. 32.]]
 ```

 元素除法：

 ```py
 # Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(x / y)
print(np.divide(x, y))
```

:
```
[[0.2        0.33333333]
 [0.42857143 0.5       ]]
[[0.2        0.33333333]
 [0.42857143 0.5       ]]
```

请注意，与 MATLAB 不同的*是，是元素乘法，而不是矩阵乘法。我们改为使用 dot 函数来计算向量的内积，将向量乘以矩阵，以及将矩阵相乘。dot 既可用作 numpy 模块中的函数，也可用作数组对象的实例方法。

```py

v = np.array([9,10])
w = np.array([11, 12])

# Inner product of vectors; both produce 219
print(v.dot(w))
print(np.dot(v, w))
```

:
```
219
219
```

您还可以使用@相当于 numpy 的dot运算符的运算符。

```py
print(v @ w)
```

:
```
219
```

Numpy 提供了许多有用的函数来对数组进行计算；最有用的方法之一是sum：

```py
x = np.array([[1,2],[3,4]])

print(np.sum(x))  # Compute sum of all elements; prints "10"
print(np.sum(x, axis=0))  # Compute sum of each column; prints "[4 6]"
print(np.sum(x, axis=1))  # Compute sum of each row; prints "[3 7]"
```

:

```
10
[4 6]
[3 7]
```


除了使用数组计算数学函数外，我们还经常需要重塑或以其他方式操作数组中的数据。 此类操作的最简单示例是转置矩阵； 要转置矩阵，只需使用数组对象的 T 属性：

```py
print(x)
print("transpose\n", x.T)
```

```
[[1 2]
 [3 4]]
transpose
 [[1 3]
 [2 4]]
 ```

 #### Broadcasting

 广播是一种强大的机制，它允许 numpy 在执行算术运算时处理不同形状的数组。我们经常有一个较小的数组和一个较大的数组，我们想多次使用较小的数组来对较大的数组执行一些操作。

例如，假设我们要向矩阵的每一行添加一个常量向量。我们可以这样做：

```py
# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)   # Create an empty matrix with the same shape as x

# Add the vector v to each row of the matrix x with an explicit loop
for i in range(4):
    y[i, :] = x[i, :] + v

print(y)
```

结果是每一行的元素都加上了 v 内对应位置的元素值

```
[[ 2  2  4]
 [ 5  5  7]
 [ 8  8 10]
 [11 11 13]]
```

由向量堆叠成为矩阵：

```py
vv = np.tile(v, (4, 1))  # Stack 4 copies of v on top of each other
print(vv)                # Prints "[[1 0 1]
                         #          [1 0 1]
                         #          [1 0 1]
                         #          [1 0 1]]"

y = x + vv  # Add x and vv elementwise
print(y)
```

```
[[1 0 1]
 [1 0 1]
 [1 0 1]
 [1 0 1]]

[[ 2  2  4]
 [ 5  5  7]
 [ 8  8 10]
 [11 11 13]]
```



