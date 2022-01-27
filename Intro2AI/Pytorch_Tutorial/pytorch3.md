#! https://zhuanlan.zhihu.com/p/462359836
# Pytorch 3. 创建一个神经网络

## 1. 导入数据

在这个教程中，我们将创建一个神经网络。首先由之前的教程，导入我们所需的数据。


```python
import torch
import torchvision
from torchvision import transforms, datasets

train = datasets.MNIST('', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

test = datasets.MNIST('', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))


trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)
```

## 2. 创建神经网络


```python
import torch.nn as nn
import torch.nn.functional as F
```

`torch.nn` 库 让我们可以访问一些有用的神经网络事物，例如各种神经网络层类型（诸如常规全连接层、卷积层（用于图像）、循环层等）。目前，我们只讨论了全连接层，所以我们现在只使用它们。

`torch.nn.functional` 库 专门为我们提供了一些我们可能不想自己编写的方便函数的访问权限。比如激活函数，优化器，分类器等。这里我们将使用 relu 作为神经元的激活函数。

为了制作我们的模型，我们将创建一个类。我们称这个类为 `net`，这个 `net` 继承自 `nn.Module` 类。

初始化后，便可以开始编写神经网络了，这里我们就是随便创建了 4 层，并称它们为：`fc1, fc2, fc3, fc4` （这里的 fc 是 fully connected 的意思，就是全连接层）。

`nn.Linear` 的作用是创建一个线性连接，即：

$$
y = wx + b
$$

并为每一次的输入和输出神经元数量赋值。第一层的 `28*28` 为图片的输入量（训练集中所有图片大小都是 `28*28` 的，这里我们将其扁平化了，实际上应该是 `1*784`），`64` 为输出层的数量。此后的每一层输入数都等于上一层的输出数。最后一层的输出数量为 `10`，这是因为我们一种有 `10` 个类别。

设置好网络的层数后，还需要设置网络的连接方式，或者说是前向传播方式 (Forward propagation)。这里我们选择的前向传播方式很简单，如函数 `forward()` 所示，层层递进。每层之间还包含一个`relu` 激活函数 `F.relu()`。 这些激活函数使我们的数据保持在 0 和 1 之间。

最后，对于输出层，我们将使用 `softmax`。`Softmax` 适用于多类问题，其中每一件事只能是一类或另一类。这意味着输出本身是一个置信度分数，加起来为 1。


```python
class Net(nn.Module):
    def __init__(self):
        # use super() to inherit the parent class functionality
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)
        
net = Net()
print(net)
```

    Net(
      (fc1): Linear(in_features=784, out_features=64, bias=True)
      (fc2): Linear(in_features=64, out_features=64, bias=True)
      (fc3): Linear(in_features=64, out_features=64, bias=True)
      (fc4): Linear(in_features=64, out_features=10, bias=True)
    )
    

至此，我们已经有了一个神经网络，我们实际上可以将数据传递给它，它会给我们一个输出。让我们来看看。让我们创建一个随机图像：


```python
X = torch.randn((28,28))
```

所以这就像我们的图像，一个 28x28 的张量（数组），值范围从 0 到 1。我们的神经网络希望它被展平，但是这样：


```python
X = X.view(-1,28*28)
```

你应该明白 28*28 的部分，但是为什么前面是 -1 呢？

我们的神经网络的任何输入和输出都应该是一组特征集。即使我们打算只传递一组特性，仍然必须将其作为特性“列表”传递。

在我们的例子中，我们真的只想要一个 1x784，我们可以这么说，但你更经常会在这些整形中使用 -1。为什么？ -1 表示“任何大小”。所以它可能是 1, 12, 92, 15295...等等。这是使该位可变的一种方便方法。在这种情况下，可变部分是我们将通过多少“样本”。

即使我们只想预测一个输入，它也需要是一个输入列表，而输出将是一个输出列表。不是一个真正的列表，它是一个张量。


```python
output = net(X)
output
```




    tensor([[-2.4276, -2.2735, -2.4012, -2.3801, -2.3522, -2.1959, -2.3241, -2.2185,
             -2.3830, -2.1177]], grad_fn=<LogSoftmaxBackward0>)



目前来说我们这个数据得到的输出对我们来说毫无价值。我们想要的是迭代我们的数据集以及进行反向传播，这就是我们将在下一个教程中介绍的内容。


- 上篇: [Pytorch 2. 数据集(Dataset)](https://zhuanlan.zhihu.com/p/462272165)
- 下篇：[]()