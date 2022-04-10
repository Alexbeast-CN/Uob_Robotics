#! https://zhuanlan.zhihu.com/p/414973930
# 机器人学课程目录

>这是来自布里斯托大学，机器人专业授课型硕士课程的开源笔记，笔记内容包括但不限于布里斯托大学的课程，其中也有斯坦福和 MIT 公开课的笔记。此专栏从2021年9月开始更新，维持更新时间预计为1年。希望可以和大家共同学习和讨论机器人相关的知识和内容。随着对在线笔记的编辑，知乎 vscode 插件的逐渐熟悉，该专栏中的笔记质量在逐渐提升。文章的格式也从 Markdown 转移到了更加方便 添加代码并运行的 jupyter notebook。

>笔记以及一些代码已有 github 托管，可以[由此链接](https://github.com/Alexbeast-CN/Uob_Robotics)访问
## 1. Robotics Systems

> 这是一门褒贬不一的课程。很多同学吐槽老师发完讲义啥也不讲，老师太好当了。但我却非常感激 Prof. Paul，原因如下：
> - 这门课程的讲义质量非常的高。讲义里有知识的传授，设置各种难度引导学生思考，最后完成实践性 Robotic Lab。
> - 这是一门研究生等级的课程，之所以叫研究生，是因为我们要掌握基础的研究能力，其中当然包括自行跟着讲义完成 Lab 的能力，以及自行上网查阅资料的能力。
> - 此外，本课程每周有一节 QA 课程，帮助同学们解决自学过程中的难题。以及一节实践课，帮助同学解决实操过程的问题。
> - Paul 对于我 As2 的帮助很大，在我提出 As2 的想法（做扫地机）后，老师第二天就给出一本算法书资料，内容精良，直击本项目的痛点。

### Assessment 1

- [Lec1. Introduction to robotic systems](https://zhuanlan.zhihu.com/p/414973593)
- [Lec2. Line following](https://zhuanlan.zhihu.com/p/416903088)
- [Lec3. Finite State Machine ](https://zhuanlan.zhihu.com/p/419237739)
- [Lec4. Obstacle Avoidance](https://zhuanlan.zhihu.com/p/423815444)
- [Lec5. Odometry 里程计](https://zhuanlan.zhihu.com/p/425715294)

### Assessment 2

第二部分的项目仓库： [Robot Navigation](https://github.com/Alexbeast-CN/Robot_navigation_webots)

- [覆盖规划算法 （Updating)](https://zhuanlan.zhihu.com/p/430612058)

## 2. Robotics Research Technology and Methods

> 这部分的内容很少的原因是
> - 大多是 saminar (研讨会)，基本都是老师讲自己的研究领域，然后我们来写一些总结作为 report。 由于比较怕老师在查重的时候查到我写的开源笔记，所以暂时不更新这部分的内容。
> - 另外一部分关于培养技能的课程过于简单，比如如何 presentation, latex, git 这些都是我早已掌握并且经常使用的技能，就不做记载了。

- [Week2. 机器人学研究的技能和方法](https://zhuanlan.zhihu.com/p/419711214)
- [论文阅读【LIO-SAM】- ENG](https://zhuanlan.zhihu.com/p/420382484)
- [如何开始学术研究（学期中的总结）](https://zhuanlan.zhihu.com/p/435485456)
  
## 3. Robotic Fundamentals

> 该部分的课程结合了多个大学的课程，包括本校，以及台大的[机器人学（一）](https://www.coursera.org/learn/robotics1/home/welcome)，英属哥伦比亚大学的[ENGR486](https://www.youtube.com/playlist?list=PLJzZfbLAMTelwaLxFXteeblbY2ytU2AxX)，国立交通大学的[机器人学理论](https://www.bilibili.com/video/BV19z4y197cf?p=16)。还有苏黎世联邦理工大学的[讲义](https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2017/RD_HS2017script.pdf)。此课程需要有一定的力学基础，有机械背景的学生学起来会相对轻松。

- [Lec1. Intro to the concept of kinematics](https://zhuanlan.zhihu.com/p/420409297)
- [Lec2. Homogeneous Transformations -1](https://zhuanlan.zhihu.com/p/423386635)
- [Lec2. Homogeneous Transformations - 2](https://zhuanlan.zhihu.com/p/426121325)
- [Lec3. Forward Kinematics](https://zhuanlan.zhihu.com/p/426994048)
- [Lec4. Inverse Kinematics in Matlab](https://zhuanlan.zhihu.com/p/430060490)
- [Lec5. 速度运动学(Velocity Kinematics)](https://zhuanlan.zhihu.com/p/445449208)
- [Lec6. 轨迹规划 (Trajectory Plan)](https://zhuanlan.zhihu.com/p/445941991)
- [机器人动力学(Dynamics) -1](https://zhuanlan.zhihu.com/p/460582634)
- [机器人动力学 (Dynamics) -2](https://zhuanlan.zhihu.com/p/460840272)

## 4. Machine Vision

> 老实说布大的计算机视觉课程~~是不错的~~（只有第一节课还不错，随后教学质量就不行了），但在我查找资料的过程中发现了一个更好的课程，斯坦福大学开设的卷积神经网络的物体识别课程，所以以后的课程我和结合布大和斯坦福的内容做笔记和总结。下面列出关于斯坦福大学的课程的资料链接：
> - [B站上的视频【熟肉】](https://www.bilibili.com/video/BV1nJ411z7fe?p=4)
> - [课程官方网站](http://cs231n.stanford.edu/) 【这里我们可以获取课程的 ppt 作业 colab文档】
> - [课程的官方 github](https://cs231n.github.io/)
> - [其他中国同学开源的作业代码](https://github.com/Halfish/cs231n)
> - [知乎上的中文笔记]( https://zhuanlan.zhihu.com/p/21930884)

- [Week1. An intro to Machine vision](https://zhuanlan.zhihu.com/p/421190397)
- [Week1. Machine Vision Tutorial from Stanford](https://zhuanlan.zhihu.com/p/422599653)
- [Opencv-python-tutorial -- 1](https://zhuanlan.zhihu.com/p/425297752)
- [Opencv-python-tutorial -- 2](https://zhuanlan.zhihu.com/p/426575079)
- [Opencv-python-tutorial -- 3](https://zhuanlan.zhihu.com/p/427681879)
- [Lec2. Image Classification](https://zhuanlan.zhihu.com/p/428291683)

> 在学习了一段时间后，发现 UWE 的 MV 与 cs231n 的内容并不一致，为了通过考试，只能换一个与学校内容相同的课程 cs131。可惜的是本课程并没有视频资料，但他们公开了课程笔记：
> - [课程教程](http://vision.stanford.edu/teaching/cs131_fall1920/syllabus.html)
> - [官方课程笔记](https://github.com/StanfordVL/CS131_notes)
> - [课程作业及答案](https://github.com/StanfordVL/CS131_release)
> - [课程中文笔记](https://github.com/zhaoxiongjun/CS131_notes_zh-CN)

- [计算机视觉基础 (Basic Knowlege of Machine Vision) -- 1](https://zhuanlan.zhihu.com/p/438616510)
- [计算机视觉基础 (Basic Knowlege of Machine Vision) -- 2](https://zhuanlan.zhihu.com/p/444536065)
- [CV3. 边缘检测 (Edge Detection)](https://zhuanlan.zhihu.com/p/446867045)
- [CV4. 特征提取 (Features) -- 1](https://zhuanlan.zhihu.com/p/448798850)
- [CV5. 特征提取 (Features) -- 2](https://zhuanlan.zhihu.com/p/449929845)

## 5. Human-Robot Interaction

> 人机交互部分与技术关系不是很大，大多数是实践与理念。因此该部分的笔记不多。

## 6. Introduction to Artificial Intelligence

> 该门课程的内容分为 Machine Learning 部分和 Deep Learning 部分。教学的方向以数据科学为主。
> 由于此部分的内容与 Stanford cs229 部分有些重复，因此重复部分会被一笔带过。

Pytorch 深度学习（布大教的是 TensorFlow）：

- [Pytorch 1. 介绍(Intro)](https://zhuanlan.zhihu.com/p/462272150) 
- [Pytorch 2. 数据集(Dataset)](https://zhuanlan.zhihu.com/p/462272165)
- [Pytorch 3. 创建一个神经网络](https://zhuanlan.zhihu.com/p/462359836)
- [Pytorch 4. 训练神经网络 (Training our Neural Network)](https://zhuanlan.zhihu.com/p/462610796)
- [Pytorch 5. 卷积神经网络(Convolutional Neural Networks)](https://zhuanlan.zhihu.com/p/463301002)
- [Pytorch 6. 使用GPU训练 (Training with GPU)](https://zhuanlan.zhihu.com/p/463450064)

数据科学：

- [DS 1. Pandas 数据分析](https://zhuanlan.zhihu.com/p/485106322)
- [DS2. Kaggle 入门 (Titanic Project Example)](https://zhuanlan.zhihu.com/p/485780305)
- [DS3. Kaggle 入门 (House Prices - Advanced Regression Techniques)](https://zhuanlan.zhihu.com/p/486448453)

算法讲解：

- [AI 1. K-means ](https://zhuanlan.zhihu.com/p/466029135)
- [AI 2. 搜索算法 (Search Algorithm)](https://zhuanlan.zhihu.com/p/467959715)
- [AI 3. A* 搜索 (A* Search)](https://zhuanlan.zhihu.com/p/469512859)


## 7. Bio-Inspire Artificial Intelligence

> 由于这门课程为 100% 的项目考核，出于时间问题，该部分的笔记可能不全。

- [BIONIC.1 神经系统(Neurla System)](https://zhuanlan.zhihu.com/p/461904298)
- [BIONIC.2 人工进化(Artificial evolution) -- 1](https://zhuanlan.zhihu.com/p/465160379)
- [BIONIC.2 人工进化(Artificial evolution) -- 2](https://zhuanlan.zhihu.com/p/465160895)

## 8. Advanced Control & Dynamics

> 此部分的笔记结合本校的课程以及[Matlab 课程](https://www.youtube.com/watch?v=hpeKrMG-WP0&t=44s&ab_channel=MATLAB) 和 [B站 DR_CAN 的视频](https://space.bilibili.com/230105574/channel/series)

> 控制论是一个强实践的课程，本门课程的理论知识并不多，但可以应用的例子非常多，因此多做题对本课程的提升会很有帮助。

- [ACnD 1. 状态空间模型 (State Space Model)](https://zhuanlan.zhihu.com/p/466790657)
- [ACnD 1. 状态空间 -- 练习](https://zhuanlan.zhihu.com/p/467039497)
- [ACnD 2. 稳定性 (Stability)](https://zhuanlan.zhihu.com/p/467088278)
- [ACnD 3. 可控性与可观测性 (Controllablity and Obeservablity)](https://zhuanlan.zhihu.com/p/467542401)
- [ACnD 4. 设计控制器 (Contorller Design)](https://zhuanlan.zhihu.com/p/470029508)
- [ACnD 5. 状态观测器 (State Observer)](https://zhuanlan.zhihu.com/p/476786539)


## 9. Machine Learning

> 该部分的内容基于 Stanford cs229。 此部分的授课方式主要是数学推导，较为硬核。更加浅显一点的课程是 cs22 Introduction to Artificial Intelligence。

- [机器学习资料整理(Machine Learning Resources)](https://zhuanlan.zhihu.com/p/450609713)
- [ML1. 线性回归(Linear Regression) -1](https://zhuanlan.zhihu.com/p/452328359)
- [ML1. 线性回归(Linear Regression) -2](https://zhuanlan.zhihu.com/p/454983290)
- [ML2. 分类器类与逻辑回归(Logistic Regression)](https://zhuanlan.zhihu.com/p/457618235)
- [ML3. 广义线性模型(Generalized Linear Models)](https://zhuanlan.zhihu.com/p/457975520)
- [ML4. 生成学习算法(Generative Learning algorithms)](https://zhuanlan.zhihu.com/p/458285940)
- [ML5. 支持向量机 (Support Vector Machine)](https://zhuanlan.zhihu.com/p/463908093)

## 10. Deep Reinforcement Learning

> 此部分的内容是为了毕业论文打基础，并非布大课程。由于前段时间为了赶 Proposal 一口气把 UCL 的 DRL 课全看完了，但没有怎么做笔记，所以先挖个坑，等有空了再慢慢把笔记都补起来。

- [DRL0. 深度强化学习资料汇总](https://zhuanlan.zhihu.com/p/472689537)
- [DRL1. 马尔可夫决策过程 (Markov Decision Processes, MDP)](https://zhuanlan.zhihu.com/p/471626124)
- [DRL 2. 动态编程 (Dynamic Programming)](https://zhuanlan.zhihu.com/p/476488801)

> 使用 OpenAI 的 baselines 和 Gym 进行的代码练习， Play With Gym

- [PWG0. 配置 Gym 的 Anaconda 环境](https://zhuanlan.zhihu.com/p/491871605)
- [PWG 1. 初识强化学习代码](https://zhuanlan.zhihu.com/p/484023706)
- [PWG2. Atari Space Invader | DQN](https://zhuanlan.zhihu.com/p/496321147)