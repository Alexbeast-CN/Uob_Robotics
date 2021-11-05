#! https://zhuanlan.zhihu.com/p/430060490
# Lec4. Inverse Kinematics in Matlab (Updating)


本章关于逆向运动学的内容，林老师讲的并不好，主要原因是方法选择。林老师选择的方法是基于人类的计算方式。但实际上，现代人应该不去手算逆向运动学。因此，本篇文章的主要参考算法为 `Jacobian Matrix`，发挥计算机的特长，用迭代的方法估计机器人的运动，这一点与 Jafari教授的思想一致。

> 资料推荐：
> - 推荐课程：[曹博士的课程：ENGR486 【A link to Youtube】](https://www.youtube.com/watch?v=CFtCqo1PV4E&list=PLJzZfbLAMTelWm74LVWgLfrsLwc_CXehR)
> - 推荐资料：[Jafari 的讲义 【百度云】](链接：https://pan.baidu.com/s/1bEuDzKkVnRlyPHvSoTIgWw)提取码：zybg  
> - 优质博客: 
>   - [V-rep学习笔记：机器人逆运动学数值解法（The Jacobian Transpose Method）](https://www.cnblogs.com/21207-iHome/p/5943167.html)
>   - [V-rep学习笔记：机器人逆运动学数值解法（The Pseudo Inverse Method）](https://www.cnblogs.com/21207-iHome/p/5944484.html)


## 1. 示例机械臂

下面的过程都将针对 `5DOF` 的 [Lynxmotion](https://www.robotshop.com/uk/lynxmotion-lss-5-dof-robotic-arm-kit.html?gclid=Cj0KCQjw5oiMBhDtARIsAJi0qk0aTqjh3t4ptAucFN1ATS-QMKLovu6-3nUB6hULnJHMPMif0c3q9VsaAtb4EALw_wcB) 机器人来求解。

机械臂样式：

![ ](./pics/lynxmotion-lss-5-dof-robotic-arm-kit-iso.png)

机械臂尺寸：

![ ](./pics/sesv2-5dof-arm-enveloppe.jpg)

其 D-H 表为：

|i  |$\alpha_{i-1}$|$a_{i-1}$|$d_i$|$\theta_i$|
|-  |-             |-        |-    |-         |
|0  |$\pi /2$      |0        |$d_1$|$\theta_1$|
|1  |0             |$L_1$    |0    |$\theta_2$|
|2  |0             |$L_2$    |0    |$\theta_3$|
|3  |$\pi /2$      |0        |0    |$\theta_4$|
|4  |0             |$L_3$    |0    |$\theta_5$|

![ ](pics/DH.png)

## 2. 多样性

在开始讲解逆向运动学之前，读者需要熟知逆向运动学的难点是什么。是对于同一点位机械臂姿态的多样性。具体如下图：

![ ](./pics/1.png)

并且随着机器人结构复杂度的增加，其解法的多样性也会增加。

## 3. Jacobian Matrix

> 建议拿起纸和笔一起推算

### 3.1 雅各比矩阵简述

我们在使用 Forward Kinematics 的时候会计算出末端执行器在世界坐标中的位置。

$$\left[
    \begin{matrix}
        x\\y\\z
    \end{matrix}
\right]=
\left[
    \begin{matrix}
        f_x(\theta_1,\theta_2...\theta_n)\\
        f_y(\theta_1,\theta_2...\theta_n)\\
        f_z(\theta_1,\theta_2...\theta_n)\\
    \end{matrix}
\right]$$

上面的矩阵式可以简化为：

$$X = f(\theta_s)$$

![ ](./pics/Jacobian.png)

要解决末端到达目标点问题：

$$\Delta X = J\Delta q$$

其中:
- $\Delta X$ 为末端执行器坐标移动的向量
- $J$ 为雅各比矩阵
- $\Delta q$ 为各个关节移动的角度

而雅克比矩阵相当于函数 $f(\theta_s)$ 的一阶导数，即线性近似。

![ ](./pics/J.png)

### 3.2 雅各比矩阵的计算


上篇：[Forward Kinematics](https://zhuanlan.zhihu.com/p/426994048)
下篇：[]()