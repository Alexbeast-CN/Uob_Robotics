#! https://zhuanlan.zhihu.com/p/423386635
# Lec2. Homogeneous Transformations

> 还是大致的看了一下Aghil Jafari教授的课程，真的让人受不了啊，这英语真的让人难以理解。所以还是弃暗投明，选择coursera上台大林沛群老师的[机器人学课程](https://www.coursera.org/learn/robotics1/home/welcome)

## 1. 刚体运动状态的描述

由于我们主要学习空间中的刚体运动，因此直接来看3D：

一般我们有3个方向(x,y,z轴)上的平移和3个方向(x,y,z轴)上的转动。共6个自由度（6DOF）。这六个自由度描述了物体在空间中姿态，如果想要描述物体的运动，只需要对物体的位移和姿态进行微分，便可以得到物体的速度和加速度等运动状态。

![ ](./pics/1.png)

### 1.1 刚体的移动

我们用向量（Vector）$\vec{p}$ 来描述 Frame B 相对于 Frame A 的移动。

![ ](pics/2.png)

比如在上面的例子中，位移向量 $\vec{p}$ 为：

$$\vec{p} = 
\left[
\begin{matrix}
P_x\\P_y\\P_z
\end{matrix}
\right]=
\left[
\begin{matrix}
10\\3\\3\\
\end{matrix}
\right]=
^{A}p_{B org}$$

### 1.2 刚体的转动

PPT 中的这张图很清楚的说明了如何利用转动矩阵来描述，Frame B 相对于 Frame A 转动的状态。我们利用一个 3X3 的矩阵来表示转动。矩阵的每一个列，代表一个轴的转动向量。矩阵用 $^{A}_{B}R$ 来表示，其中上标 `A` 为 `base` 基坐标，下坐 `B` 为 `Target`目标坐标。
 
![ ](pics/3.png)

每一个列向量实则为 `B` 中的基向量$\vec{x},\vec{y},\vec{z}$ 在 `A` 上的投影。比如下面的例题： 

![ ](pics/4.png)

### 1.3 旋转矩阵

**特性：**

- <center>矩阵列向量长度为1</center>
- <center>列向量之间两两垂直</center>
- $$^{A}_{B}R = ^{B}_{A}R^T = ^{B}_{A}R^{-1}$$
- $$^{A}_{B}R ^{B}_{A}R^T = I_3$$
- $$Det(R) = 1$$
- $$Det(R^{T}) = -1$$

旋转矩阵除了表述 Frame 之间的姿态关系外，也可以用于转换向量之间的坐标。

$$^{A}P = ^{A}_{B}R^{B}P$$

> 一定注意，是左乘旋转矩阵！

**计算方法：**

对于Frame A 到 Frame B 的旋转矩阵 R：

$$_{B}^{A}R = 
    \left[
    \begin{matrix}
        |&|&|\\
        ^{A}\hat{X}_B&^{A}\hat{Y}_B&^{A}\hat{Z}_B\\
        |&|&|\\
    \end{matrix}
    \right]$$

- 绕 Z 轴旋转：
    $$R_z(\theta)=
    \left[
    \begin{matrix}
        cos\theta & -sin\theta & 0\\
        sin\theta & cos\theta & 0\\
        0 & 0 & 1
    \end{matrix}
    \right]=
     \left[
    \begin{matrix}
        c\theta & -s\theta & 0\\
        s\theta & c\theta & 0\\
        0 & 0 & 1
    \end{matrix}
    \right]$$

- 绕 X 轴旋转：
    $$R_x(\theta)=
    \left[
    \begin{matrix}
        1&0&0\\
        0&cos\theta&-sin\theta\\
        0&sin\theta&cos\theta\\
    \end{matrix}
    \right]=
     \left[
    \begin{matrix}
        1&0&0\\
        0&c\theta&-s\theta\\
        0&s\theta&c\theta\\
    \end{matrix}
    \right]$$

- 绕 Y 轴旋转：
    $$R_y(\theta)=
    \left[
    \begin{matrix}
        cos\theta&0&sin\theta\\
        0&1&0\\
        -sin\theta&0&cos\theta\\
    \end{matrix}
    \right]=
     \left[
    \begin{matrix}
        c\theta&0&s\theta\\
        0&1&0\\
        -s\theta&0&c\theta\\
    \end{matrix}
    \right]$$

> 逆时针的旋转方向为正

> 在 Matlab 2021b 及以上版本中可以使用 rotx(), roty(), rotz() 来计算旋转矩阵。

![ ](pics/5.png)

以上例题，在 `Matlab Online` 中就可以使用以下代码来求解：

```Matlab
P = [0;1;1.732]
R = rotx(30)
Pnew = R*P
```

Output:
![ ](pics/6.png)

 