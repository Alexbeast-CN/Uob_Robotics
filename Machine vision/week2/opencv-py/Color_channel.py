import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('Machine vision\week2\pics\cats.jpg')
cv.imshow('cats',img)

blank = np.zeros(img.shape[:2], dtype='uint8')

b,g,r = cv.split(img)

# 三个通道的图片展示出来都是色度图，但三者之间有所区别
# 黑色的地方表示这里的所含的通道的颜色少，白色反之。
# 比如蓝天在蓝色通道图中看来来就比较白。
# cv.imshow('Blue',b)
# cv.imshow('Green',g)
# cv.imshow('Red',r)

# 将3个通道合成彩色图片
merged = cv.merge([b,g,r])
# cv.imshow('merged image',merged)

# 将单个颜色通道与黑色背景融合
blue = cv.merge([b,blank,blank])
green = cv.merge([blank,g,blank])
red = cv.merge([blank,blank,r])

cv.imshow('blue',blue)
cv.imshow('green',green)
cv.imshow('red',red)


cv.waitKey(0)