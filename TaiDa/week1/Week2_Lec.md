#! https://zhuanlan.zhihu.com/p/426121325
# Lec2. Homogeneous Transformations - 2

通过之前的学习，我们知道了刚体的质心状态可以通过坐标轴 (frame) 上的点来表示。而坐标点的变换由可以分解为旋转和平移两种运动方式。

平移：

$$^{A}R_{B org} = 
\left[
\begin{matrix}
P_{B x} \\
P_{B y} \\
P_{B z} \\
\end{matrix}
\right]$$

{B} frame 对 {A} frame 的平移

转动：

$$_{B}^{A}R = 
\left[
\begin{matrix}
    |&|&|\\
    ^{A}\hat{X}_B&^{A}\hat{Y}_B&^{A}\hat{Z}_B\\
    |&|&|\\
\end{matrix}
\right]$$

整合后：

$$\left\{
    B
\right\}=
\left\{
    ^{A}_{B}R, ^{A}P_{B org}
\right\}$$

考虑到将两者整合为同一个矩阵后的运算方便性，对其使用增广，从而产生一个 (4x4) 的齐次矩阵。

Homogeneous transformation matrix:

$$\left[
    \begin{matrix}
        ^{A}_{B}R \  _{3\times3} & ^{A}P_{B org} \ _{3\times1}\\
        0\ 0\ 0 & 1\\
    \end{matrix}
\right]=
\left[
    \begin{matrix}
        |&|&|&|\\
        ^{A}\hat{X}_B&^{A}\hat{Y}_B&^{A}\hat{Z}_B&^{A}P_{B org}\\
        |&|&|&|\\
        0&0&0&1\\
    \end{matrix}
\right]=
^{A}_{B}T$$

有了 Transformation Matrix, T 后， 我们就可以使用一个矩阵来表示坐标轴之间的变换了。如下图，当我们知道向量 $P$ 对于 frame {A} 的向量为 $^{A}P$ 和{A} 到 {B} 的变换方式后，便可以求出 $P$ 再 frame {B} 中的向量 {^{B}P}。

![ ](../week2/pics/1.png)

- 仅有移动时：

$$^{A}P\ _{3\times1} = ^{B}P\ _{3\times1} + ^{A}P_{B org}\ _{3\times1}$$

即简单的向量相加。其齐次矩阵形式为：

$$\left[
    \begin{matrix}
        ^{A}P \\
        1\\
    \end{matrix}
\right]=
\left[
    \begin{matrix}
        I_{3\times3}&^{A}P_{B org}\ _{3\times1}\\
        0\quad 0\quad 0& 1\\
    \end{matrix}
\right]
\left[
    \begin{matrix}
        ^{B}P \\
        1\\
    \end{matrix}
\right]=
\left[
    \begin{matrix}
        ^{B}P + ^{A}P_{B org} \\
        1\\
    \end{matrix}
\right]$$

- 仅有旋转时：

$$^{A}P\ _{3\times1} = ^{A}_{B}R_{3\times3} \ ^{B}P\ _{3\times1} $$

> 注意：是左乘

$$\left[
    \begin{matrix}
        ^{A}P \\
        1\\
    \end{matrix}
\right]=
\left[
    \begin{matrix}
        ^{A}_{B}R_{3\times3}&\begin{matrix}
            0\\0\\0
        \end{matrix}\\
        0\quad 0\quad 0& 1\\
    \end{matrix}
\right]
\left[
    \begin{matrix}
        ^{B}P \\
        1\\
    \end{matrix}
\right]=
\left[
    \begin{matrix}
        ^{A}_{B}R_{3\times3} \ ^{B}P\ _{3\times1} \\
        1\\
    \end{matrix}
\right]$$

- 当移动和旋转复合时：

$$^{A}P\ _{3\times1} = ^{A}_{B}R_{3\times3} \ ^{B}P\ _{3\times1} + ^{A}P_{B org}\ _{3\times1}$$

> 注意我们一般先让其旋转，后平移。

$$\left[
    \begin{matrix}
        ^{A}P \\
        1\\
    \end{matrix}
\right]=
\left[
    \begin{matrix}
        ^{A}_{B}R_{3\times3}&^{A}P_{B org}\ _{3\times1}\\
        0\quad 0\quad 0& 1\\
    \end{matrix}
\right]
\left[
    \begin{matrix}
        ^{B}P \\
        1\\
    \end{matrix}
\right]=
\left[
    \begin{matrix}
        ^{A}_{B}R \ ^{B}P + ^{A}P_{B org}\ \\
        1\\
    \end{matrix}
\right]$$

例题1：

![ ](../week2/pics/2.png)

使用 `MATLAB ONLINE` 进行运算：

```matlab
PB = [3;7;0;1];
PAB = [10;5;0];
XAB = [sqrt(3)/2;1/2;0];
YAB = [-1/2;sqrt(3)/2;0];
ZAB = [0;0;1];

RAB = [XAB YAB ZAB]
TAB = [RAB PAB;0 0 0 1]
PA = TAB*PB
```

![ ](../week2/pics/3.png)

> 但是一定要注意，我们计算变换矩阵 `T` 都是基于先旋转后平移的方法。如果先平移后转动就会出现不一样的结果。

- 先旋转后平移：
$$P_{2} = R \ P_{1}\ + ^{A}Q$$

- 先平移后转动：
![ ](../week2/pics/4.png)

矩阵形式为：

$$P_2 =
\left[
    \begin{matrix}
        R(\theta)&\begin{matrix}
            0\\0\\0
        \end{matrix}\\
        0\quad 0\quad 0& 1\\
    \end{matrix}
\right]
\left[
    \begin{matrix}
        I_{3\times3}&^{A}Q\\
        0\quad 0\quad 0& 1\\
    \end{matrix}
\right]
$$

因为运动是相对的，$^{A}_{B}𝑇$当Operator时对向量（或点）进行移动或转动的操作，也可以想成是对frame进行「反向」的移动或转动的操作。

![ ](../week2/pics/5.png)
