#! https://zhuanlan.zhihu.com/p/433219993
# 机器人领域论文阅读 -- 2

## 1. 形态计算：软体隐藏的超能力 --  Helmut Hauser

受自然启发的形态学可以成为构建智能机器的关键原则。 即使没有控制器、传感器或大脑，事物也只能通过形态对不断变化的环境做出智能反应。 挑战在于如何建立一个可以改善与环境的互动并促进学习和控制的身体。 一些基于该理论的项目，如蜘蛛网、传感皮肤和水母机器人。 该理论的下一个层次是找出形态学中的逻辑功能。 该领域将利用化学和生物知识来构建能够合乎逻辑地进行运动的机器人。

<div  align="center"> 
<img src="week4/pics/6.png" width = "500"  alt="Fig8. The hierarchy structure of explainable learning" align=center />
</div>
<center>Fig1. The key structure of morphological robot</center>

## 2. ORB-SLAM3：用于视觉、视觉惯性和多地图 SLAM 的准确开源库 -- Carlos Campos

ORB-SLAM3 是一个开源的视觉惯性 SLAM 框架。 ORB-SALM3 的映射精度比旧的 ORB-SLAM2 好 2-5 倍，被证明是最好的视觉 SLAM 框架。 他们新颖地提出了一种多地图系统，可以在要素丢失时启动新地图。 新地图可以与之前的地图无缝融合，这使得这个 ORB-SLAM3 更加健壮。 下图显示了来自相机的所有数据，IMU 将通过跟踪线程来决定关键帧，从而形成本地地图。 然后经过地点识别、环路校正和地图合并，它们成为整个地图的一部分。

<div  align="center"> 
<img src="week3/3.png" width = "500"  alt="Fig8. The hierarchy structure of explainable learning" align=center />
</div>
<center> Fig2. Main system components of ORB-SLAM3</center>

