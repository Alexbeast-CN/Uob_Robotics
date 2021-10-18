#! https://zhuanlan.zhihu.com/p/422599653
# Week1. Machine Vision Tutorial （文中所提及的 Colab 在中国无法正常使用）

> 本篇笔记的内容基于[斯坦福的 Colab Notes](https://colab.research.google.com/github/cs231n/cs231n.github.io/blob/master/python-colab.ipynb#scrollTo=dzNng6vCL9eP)

> colab的本质是一个 web 版的 python ide + Markdown 编辑器。而从个人习惯来说，我更喜欢使用 Vscode 作为我的编辑器，但可惜的是我的电脑（ThinkPad X1C）没有强大的 GPU 所以最后关于卷积神经网络的部分依然会在 Colab 中完成。

## Python Tutorial With Google Colab

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

Print Output:
```
<string>:1: UserWarning: The NumPy module was reloaded (imported a second time). This can in some cases result in small but subtle issues and is discouraged.
4.5.3
1.21.2
[1, 1, 2, 3, 6, 8, 10]
```

### Containers

#### List

A list is the Python equivalent of an array, but is resizeable and can contain elements of different types:

```py
xs = [3, 1, 2]   # Create a list
print(xs, xs[2])
# -1 means the end of list
print(xs[-1])     # Negative indices count from the end of the list; prints "2"
```


Print Output:
```
[3, 1, 2] 2
2
```

```py
xs[2] = 'foo'    # Lists can contain elements of different types
print(xs)
```

Print Output:
```
[3, 1, 'foo']
```

```py
xs.append('bar') # Add a new element to the end of the list
print(xs)
```

Print Output:
```
[3, 1, 'foo', 'bar']
```

```py
x = xs.pop()     # Remove and return the last element of the list
print(x, xs)
```

Print Output:
```
bar [3, 1, 'foo']
```

For more reference check [Python Documents](https://docs.python.org/3.7/)

