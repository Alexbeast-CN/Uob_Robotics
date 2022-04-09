#! https://zhuanlan.zhihu.com/p/472689537
# DRL 0. 深度强化学习资料汇总

> 以下链接均导航指 Youtube，对无法科学上网的同学深感抱歉。

## 1. 理论课程

- 斯坦福大学的 [cs234](https://www.bilibili.com/video/BV1sb411s7eQ?from=search&seid=14467709922277911537&spm_id_from=333.337.0.0)。可惜的是课程的官方资料被关闭了，因此只找到了一些非官方的资料，以供参考。为了方便国内的同学查看，一部分资料已存入百度网盘。
  - [PPT](https://pan.baidu.com/s/1h9YNIQ6QeAmLU8N4IOOt9g) 提取码：vmno
  - [课程中文讲义](https://github.com/apachecn/stanford-cs234-notes-zh)
- UC, Berckley + OpenAI 的深度强化学习网课 [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures)
  - [习题资料](https://github.com/simonmeister/deep-rl-bootcamp)
- UC, Berckley 校内自己的强化学习网课 [CS 285](https://www.youtube.com/playlist?list=PL_iWQOsE6TfXxKgI1GgyV1B_Xa0DxE5eH)
- 书籍 [Reinforcement learning an introduction](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)
  - [code from the book](http://incompleteideas.net/book/code/code2nd.html)
- 一个来自程[序员信息网](https://www.i4k.xyz/) 的非常好的资料页 [强化学习的学习之路](https://www.i4k.xyz/article/zyh19980527/112592306)
- [RL baseline](https://stable-baselines.readthedocs.io/en/master/)
- DeepMind 和 UCL 一起开设的网课 [2021 DeepMind x UCL RL Lecture Series](https://www.youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm)
  - Hado van Hasselt 老师的个人站[Hado van Hasselt](https://hadovanhasselt.com/)
- DeepMind 大佬 David Silve 2015年的 RL 网课 [RL Course by David Silver](https://www.youtube.com/watch?v=2pWv7GOvuf0&t=16s&ab_channel=DeepMind)
- CMU [Deep Reinforcement Learning and Control](https://www.andrew.cmu.edu/course/10-703/) 这个课程是我在读论文[Deep Reinforcement Learning for AutonomousDriving](https://arxiv.org/abs/1811.11329)的时候偶然发现的，一篇顶会论文居然是这门课程的课设。实在了不起！

## 2. 代码实践

在网课中个人比较喜欢 Stanford 和 UCL 的。两个原因，一个是因为我之前的人工智能可能都是跟着 Stanford 网课学习的，有一脉相承的感觉。UCL 是因为我自己是英国留学生，UCL 是英国学校，且课程很新。另一个原因是 Hado van Hasselt 大佬真的牛逼

关于代码实践方面，我专门做了一个子专栏 [Play With Gym](https://zhuanlan.zhihu.com/p/491871605)。这里我主要使用 OpenAI 的 [Gym](https://gym.openai.com/) 环境作为来训练强化学习的 Agent。

与实践相关的课程我目前发现了以下的两个：

华盛顿大学 [Jeff Heaton](https://sites.wustl.edu/jeffheaton/)教授的 课程 [Applications of Deep Neural Networks](https://sites.wustl.edu/jeffheaton/t81-558/)。
- 该课程视频位于 [Youtube | Course on the Application of Deep Neural Networks ](https://www.youtube.com/watch?v=sRy26qWejOI&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN&ab_channel=JeffHeaton)。
- 该课程的代码仓库位于 [Github | t81_558_deep_learning](https://github.com/jeffheaton/t81_558_deep_learning)。
- 该课程的 Colab 链接位于：[Colab | AI_GYM.ipynb](https://colab.research.google.com/github/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_12_01_ai_gym.ipynb#scrollTo=dnID4yguIeX7)


此外还有 MIT 大佬 PhD [Alexander Amini](https://www.mit.edu/~amini/) 开设的 [Introduction to Deep Learning](http://introtodeeplearning.com/)
- 课程的视频位于 [Youtube | MIT Introduction to Deep Learning | 6.S191](https://www.youtube.com/watch?v=7sB052Pz0sQ&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&ab_channel=AlexanderAmini)
- 课程的代码仓库位于 [Github | Aamini/introtodeeplearning](https://github.com/aamini/introtodeeplearning)
- Colab 笔记链接为 [Colab | Laboratory 3: Reinforcement Learning](https://colab.research.google.com/github/aamini/introtodeeplearning/blob/master/lab3/solutions/RL_Solution.ipynb)

另外关于强化学习的算法库我找到了以下几个：

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/)
- OpenAI [Spinning Up](https://spinningup.openai.com/en/latest/)
- Ray [RLlib](https://docs.ray.io/en/master/rllib/index.html)

> 最为初学者我推荐使用 Baselines3，但也有很多人使用 Ray 的 RLlib，这个就看个人了。
