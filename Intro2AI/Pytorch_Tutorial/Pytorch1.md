#! https://zhuanlan.zhihu.com/p/462272150
# Pytorch 1. 介绍(Intro)

> 此教程跟随 youtuber: Sentdex 的 [Pytorch 教程](https://www.youtube.com/watch?v=ixathu7U-LQ&list=PLQVvvaa0QuDdeMyHEYc0gxFpYwHY2Qfdh&index=3&ab_channel=sentdex) 创作。其英文版的链接如下：
> https://pythonprogramming.net/introduction-deep-learning-neural-network-pytorch/

## 1. 环境

通常来说，深度学习需要有 GPU 加速训练神经网络。所以对于本地有不错 GPU 的选手(RTX 2070 以上)可以选择在本地训练一个神经网络。如果你像我没有 GPU 的话，可以选择使用 GPU 云服务器。不想花钱的话可以使用 Google Colab 或者 Kaggle 白嫖 GPU 。但这些平台通常都是限时的，所以对于大型的程序而言，训练到一半被平台踢出来真的很不爽，之前的东西都白训练了 (Colab Pro 也没啥用)。所以如果是要做大型的项目，最好还是花钱租用 GPU 云服务器，平台很多，价格都差不多的贵。。。

## 2. 什么是 Pytorch

Pytorch 库与其他深度学习库一样，实际上只是一个对张量进行操作的库。张量用一个不太成熟的理解，就是矩阵，而 Pytorch 就是一个可以调用 GPU 的 numpy 库。

### 2.1 什么是张量 (Tensor)

通过下面的代码我们就可以创建一些 Tensor，他们看上去就和向量差不多。


```python
# if you don't have pytorch, you can install it by:
# pip install torch torchvision

# pytorch is like a numpy running on gpu
import torch

# We can treat tensor like an array for now
# Now let's do some math
x = torch.tensor([5,3])
y = torch.tensor([2,1])

print(x*y)
```

    tensor([10,  3])
    

下面会继续探索一些 pytorch 的其他基础功能。这些功能大多都和 numpy 的矩阵操作类似。


```python
x = torch.zeros([2,5])
print(x)
x.shape
```

    tensor([[0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]])
    




    torch.Size([2, 5])




```python
y = torch.rand([2,5])
y
```




    tensor([[0.6910, 0.3738, 0.5051, 0.0293, 0.1903],
            [0.5467, 0.8598, 0.3174, 0.6280, 0.4812]])



对于矩阵的变形，比如将一个 $2\times 2$ 的矩阵变成一个 $1 \times 4$ 的矩阵。在 numpy 中我们使用的函数是 reshape()，但 pytorch 使用 view() 来操作。


```python
y = y.view([1,10])
y
```




    tensor([[0.6910, 0.3738, 0.5051, 0.0293, 0.1903, 0.5467, 0.8598, 0.3174, 0.6280,
             0.4812]])

- 下篇：[Pytorch 2. 数据集(Dataset)](https://zhuanlan.zhihu.com/p/462272165)