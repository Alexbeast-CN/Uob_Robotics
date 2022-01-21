# 机器人动力学(Dynamics)

## 1. 简介

机器人的动力学模型解释了当力作用与系统时会产生的影响。这对于机器人的仿真与控制都是重要的。动力学机器人与运动学机器人在应用上的对比就是工业机器人与协助机器人。通常来说，力控的存在，协作机器人做视为安全的机器人，而工业机器人通常较为危险，人类不可以靠近。

![工业机器人](pics/v2-feacca79463c7c9835485139430bcd59_b.jpg)

<center>

工业机器人

</center>

![协作机器人](./pics/Uni-of-Sheffield-robot.jpg)

<center>

协作机器人

</center>

由理论力学的基础可以得到，对于一个固定底座的多轴机器人，其动力学方程可以写成：

$$
M(q)\ddot{q} + b(q,\dot{q}) + q(q) = \tau + J_c(q)^TF_c
$$

其中：

- $M(q) \in \mathbb{R}^{n_q\times n_q}$  -- 广义质量矩阵
- $q,\dot{q},\ddot{q} \in \mathbb{R}^{n_q}$  -- 广义位置、速度和加速度向量
- $b(q,\dot{q}) \in \mathbb{R}^{n_q}$ -- 离心力项
- $g(q) \in \mathbb{R}^{n_q}$ -- 引力项
- $\tau \in \mathbb{R}^{n_q}$ -- 外部广义力
- $F_c \in \mathbb{R}^{n_c}$ -- 外部笛卡尔力
- $J_c(q) \in \mathbb{R}^{n_c \times n_q}$ -- 对应于外力的几何雅可比行列式

这上面的公式可以给我们一个大体的印象，就是说动力学是与速度，加速度，力，和扭矩有关的。在计算的时候，一般会用到两种计算方法：**拉格朗日法(Lagrange Method)** 和 **牛顿欧拉法 (Newton-Eulermethod)**

## 2. 