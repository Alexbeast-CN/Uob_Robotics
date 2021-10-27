import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('Machine vision\week2\pics/1.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

# Simple Thresholding
# If pixel value is larger than 125, it will be set to 255.
# If lower than 125, it will be set 0.
# 最后一个参数用于控制图片呈现的样式
# 图像编程了一个二进制非黑即白的图像
threshold, thresh = cv.threshold(gray,125,255,cv.THRESH_BINARY)
cv.imshow('Simple_threshold',thresh)
cv.imwrite('Machine vision\week3\pics/Simple_threshold.jpg',thresh)

# Inver Thresholding
# 将之前图片的黑白颠倒
threshold, thresh_inv = cv.threshold(gray,125,255,cv.THRESH_BINARY_INV)
cv.imshow('Simple_threshold_inv',thresh_inv)
cv.imwrite('Machine vision\week3\pics/Simple_threshold_inv.jpg',thresh_inv)

# Adaptive Thresholding
adapt_thresh = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,3)
cv.imshow('Simple_threshold_inv',adapt_thresh)
cv.imwrite('Machine vision\week3\pics/adapt_thresh.jpg',adapt_thresh)

cv.waitKey(0)