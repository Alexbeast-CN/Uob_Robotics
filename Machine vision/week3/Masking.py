import cv2 as cv
import numpy as np

img = cv.imread('Machine vision\week2\pics/1.jpg')
cv.imshow('org',img)
cv.imwrite('Machine vision\week3\pics/img.jpg',img)

blank = np.zeros(img.shape[:2],dtype='uint8')
mask = cv.circle(blank, (img.shape[1]//2,img.shape[0]//2),200,255,-1)
cv.imshow('mask',mask)
cv.imwrite('Machine vision\week3\pics/mask.jpg',mask)

# 注意这里传的前两个参数都是原图
masked = cv.bitwise_and(img,img,mask=mask)
cv.imshow('masked',masked)
cv.imwrite('Machine vision\week3\pics/masked.jpg',masked)

cv.waitKey(0)