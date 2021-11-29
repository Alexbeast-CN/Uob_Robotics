#! https://zhuanlan.zhihu.com/p/435485456
# 如何开始学术研究（学期中的总结）

> 以机器人学专业中的[Robotic System 项目](https://github.com/Alexbeast-CN/Robot_navigation_webots) 为例的学术研究介绍

## 1. 培养学术技能

> 做学术不是闭门造车，我们需要培养学术技能。日常聊天的交流方式一般无法深度而细致的进行学术探讨，因此作为一个研究生需要掌握：写论文，读论文，Presentation 等方式与世界各地的学者进行思想交流。

### 1.1 读论文

> 作为一个学者，我们必须对于自己研究领域的知识有一定的储备，我们不一定需要知道所有自己领域的知识，但一定要知道这个领域里最新的研究是什么。这样以为着，作为学者，我们必须保持阅读论文的习惯，才能让我们的思想常青。

#### 1.1.1 论文的种类：

- 会议论文
  - 会议旨在快速交流思想，因此他们的论文相对较短，通常有六页长。
- 期刊论文
  - 期刊论文比会议更长，周转速度更慢。 因此，许多想法首先出现在会议上，然后通过添加更彻底的解释或评估，然后转化为期刊论文。
- 书籍
  - 教科书
    - 教科书里通常会为学术提供最基础的研究方法。
  - 合作书籍
    - 由多个作者共同完成的书籍，每章的内容来自不同的学者。
  - 专著
    - 一般专著是一个 PhD 的所有研究内容的合集，一人独自完成。


#### 1.1.2 如何查找论文

> 开个玩笑，“知网是什么？”

作为在校生的我们可以利用院校登录各大论文机构的网站，比如：

- [IEEE](https://ieeexplore.ieee.org/Xplore/home.jsp)
- [ScienceDirect](https://www.sciencedirect.com/)
- [google scholar](https://scholar.google.com/)

> 对于已经毕业的人来说，并且公司没有账号的朋友来说，没有那么合法的 sci-hub 也许可以帮助你。

#### 1.1.3 如何读论文

这个不是三言两语能够说清楚的。大多数人在刚刚接触到论文的时候都会觉得论文很难读，读不下去。这都是正常的，尤其是英文论文，对于英语不好的同学来说，简直是噩梦（所以学好英语很重要）。那么关于如何论论文有什么好的建议吗？有：

- [Academic Reading - from UoB Study Skills](https://www.ole.bris.ac.uk/bbcswebdav/pid-5855635-dt-content-rid-10970147_2/courses/Study_Skills/academic-reading/index.html)
- [Guide to Academic Reading - from Towards Data Science](https://towardsdatascience.com/guide-to-reading-academic-research-papers-c69c21619de6)
- [How to Read a Scientific Paper - from Science Magazine](https://www.science.org/content/article/how-read-scientific-paper-rev2)
- [Reading and Researching  - from the Royal Literary Fund](https://www.rlf.org.uk/resources/how-not-to-read/)
- [相关的 PDF 文件 - 提取码：lltw ](https://pan.baidu.com/s/14y2g5iLOHscFKan6Dg5YUg)

> 英文阅读有障碍又苦恼 pdf 文件无法翻译的同学，可以尝试[pdftranlator](https://github.com/axipo/pdfTranslator/releases)。这是我最喜欢的翻译插件，没有之一。

知乎上也有不少相关的建议，大家自行搜索，我就不再推荐了。

#### 1.1.4 养成良好的文件整理习惯

对于自己已经读过的论文，或者是大概看过，感觉没有用的论文。我们都应该整理好，方便以后查阅（好记性不如烂笔头）。这里推荐一些论文整理软件：

- [EndNote](https://endnote.com/) （要花钱，但也有免费的）
- [Mendeley](https://www.mendeley.com/download-reference-manager/windows) （学生免费）

### 1.2 写论文

#### 1.2.1 Latex

大多数人使用 Word 在写文章，为了获得美观的数学公式和文章的排版更加的符合规范等等，使用 `Latex` 来撰写文章变得非常有必要。但 `Latex` 是一个较为复杂的格式，在使用之前最好进行一段时间的学习，这里我推荐先观看一个短片来了解一下[什么是 Latex](https://www.bilibili.com/video/BV11h41127FD?from=search&seid=4578307811589895707&spm_id_from=333.337.0.0)。之后，我们只需要下载好学校提供的 `Latex` 模板，以及使用 [Overleaf](https://www.overleaf.com/)，便可以使用 `Latex` 进行创作了。如果你还不知什么是 `Overleaf` 可以观看这个视频：[Overleaf 入门](https://www.bilibili.com/video/BV15v411A7xw/?spm_id_from=333.788.recommend_more_video.1)。

剩下的大多数时间就在使用谷歌/百度搜索 `Latex` 的一些细节用法。

#### 1.2.2 论文内容

到底怎么样的文章才能算是一篇论文呢？一篇论文（以 Robotics System 的项目为例）

1. 首先在我们找到一个有趣的提案后，我们需要确定这个项目的目标和对象(Aims & Objectives)是什么？
2. 其次需要提出一个假设(hypothesis) 。比如，如果我利用覆盖算法做扫地机器人，这会大大提升扫地机器人的性能。
3. 在有了这个猜想后，我们需要一个衡量标准(Metric)，它可以是一个对比的对象，用来衡量性能到底提升了多少。比如我的新算法与目前常用的算法相比在打扫同一个房间时，路径可以缩短多少？
4. 然后，我们要把自己的实验方法(Experiment Method)一步一步，详细的写出来。而且要保证别人在看了我们的论文后，使用相同的方法，也可以做出相同的实验效果。这样科学才可以有效传播。
5. 完成实验之后，我们需要展示实验数据并且分析得到的数据。是否达到了我们的预期，如果是的，或者不是的，其印象因素有哪些？以及是否实验结果能否再有所提升，可以如何提升？
6. 最后将整篇文章做一个总结，以及概述。

### 1.3 实操能力

> 机器人学是一个结合了机械工程、电气电子工程、计算机科学、人工智能等领域的复杂学科。但从目前的趋势来看，大多数的研究集中在计算机科学和人工智能领域。因此接下来的内容主要讲解计算机和人工智能领域获取知识的渠道。

#### 1.3.1 代码问题

首先代码是一个强实践的课程，不是说光看书就可以学会的。因此，在刚接触一个语言的时候，最好的方式是先跟着一个教程做，把最最基础的内容学会之后，用一个个简单的项目来练手。多看别人写的代码，多看源码，见的多了自然也就会用了。下面我会更加详细的解释，我总结的学习代码过程，以 `C++` 为例子:

1. 首先，要掌握 `C++` 基础，可以先看一下[侯捷大佬的 `C++` 课程](https://www.youtube.com/watch?v=2S-tJaPKFdQ&list=PL-X74YXt4LVZ137kKM5dNfCIC4tsScerb&ab_channel=%E5%90%AC%E6%B6%9B%E9%98%81)。很不幸的是 B站下架了所有侯捷的课程，所以只能提供 Youtube 的链接
2. 对于 `C++` 有一定了解后，可以在 `GitHub` 上找一些项目，看别人的代码是怎么写的。
3. 在使用 `C++` 时遇到问题有以下几个途径解决问题：
   - 第一步在谷歌上输入问题，一般来说谷歌给的前几条内容就可以给出答案，如果没有，看下面的几条。
   - 如果是基础问题，如果语法问题推荐使用 [C++ Refernce 中文](http://c.biancheng.net/cplus/), [C++ Refernce 官方](https://www.cplusplus.com/reference/clibrary/)
   - 如果是一些奇奇怪怪的，自己也不知道怎么回事的问题，可以在 [Stack Overflow](https://stackoverflow.com/) 上搜索。
   - 如果是宏观一点的问题，比如连编程思路都没有，可以先在 [Github](https://github.com/) 搜索有没有人做过类似的项目。

#### 1.3.2 Linux

对于我的专业来说，推荐使用的系统的 Linux。这是一个机器人普遍使用的系统，所以非常有必要学习。初学者推荐先使用虚拟机，熟练之后再把自己的系统安装在一个移动硬盘里，最后需要大量使用的时候，再把自己使用的电脑系统换成 Linux。至于每一步要怎么做，自行谷歌。在有了一个 Linux 系统之后，一般人大约需要花费 2-3 周的时间去学习和适应这个系统。因为它的操作逻辑与 `Windods` 或者 `Mac` 不太相同，具体的说是更加复杂。不过一但掌握了 Linux 系统，你就会感叹开发离不开 Linux 了。

#### 1.3.3 合作

一个伟大的项目通常都是多人合作完成的。这里依然以 [Robotics System 的项目](https://github.com/Alexbeast-CN/Robot_navigation_webots)，来谈谈代码类的项目该如何合作。首先创建一个 Github 仓库，将合作者作为 `Contributor` 邀请进入仓库，这样双方在 `Git clone` 这个项目的仓库到本地后，每个人的每一次更新都会同步到这个仓库，然后同步到对方的电脑上。这一篇[CSDN上的博客](https://blog.csdn.net/dengdengda/article/details/50903176) 介绍的方法与我使用的大致相同。

下一篇我将介绍第二部，介绍学术思想。