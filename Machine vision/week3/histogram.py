import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('Machine vision\week2\pics/1.jpg')

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)
cv.imwrite('Machine vision\week3\pics/gray.jpg',gray)

#创建图片的遮罩
blank = np.zeros(img.shape[:2],dtype='uint8')
circle = cv.circle(blank,(img.shape[1]//2,img.shape[0]//2),200,255,-1)
mask = cv.bitwise_and(gray,gray,mask=circle)
cv.imshow('mask',mask)
cv.imwrite('Machine vision\week3\pics/mask.jpg',mask)

# Grayscale histogram
# 这里是画出灰度图的像素集中度，[0,256] 为从黑到白
gray_hist = cv.calcHist([gray],[0],None,[256],[0,256])
mask_hist = cv.calcHist([mask],[0],None,[256],[0,256])

# 使用 plt 画出直方图
# plt.figure()
# plt.subplot(2,1,1)
# plt.title('Histogram of Gray')
# plt.xlabel('Bin')
# plt.ylabel('# of pixels')
# plt.plot(gray_hist)
# plt.xlim([0,256])

# plt.subplot(2,1,2)
# plt.title('Histogram of Mask')
# plt.xlabel('Bin')
# plt.ylabel('# of pixels')
# plt.plot(mask_hist)
# plt.xlim([0,256])
# plt.show()

plt.figure()
plt.title('Histogram of Color')
plt.xlabel('Bin')
plt.ylabel('# of pixels')
colors = ('b', 'g', 'r')
for i,col in enumerate(colors):
    hist = cv.calcHist([img],[i],None,[236],[0,256])
    plt.plot(hist,color=col)
    plt.xlim([0,256])
plt.show()

cv.waitKey(0)