## 4. 尺度不变关键点检测(Scale Invariant Keypoint Detection)

### 4.1 动机

对于 Harris Detector 来说，图片的尺寸变化对于检测有着很大的影响。比如下图是一个拐角的放大图，利用小号的窗口，将无法检测到这张放大后的拐角。

![ ](pics/invariant.png)

### 4.2 方法