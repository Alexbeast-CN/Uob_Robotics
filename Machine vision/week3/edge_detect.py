import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('Machine vision\week3\pics/bristol.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)
cv.imwrite('Machine vision\week3\pics/bristol_gray.jpg',gray)

# Laplcaian
lap = cv.Laplacian(gray,cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplcaian',lap)
cv.imwrite('Machine vision\week3\pics/lap.jpg',lap)

# Sobel

sobelx = cv.Sobel(gray,cv.CV_64F,1,0)
sobely = cv.Sobel(gray,cv.CV_64F,0,1)

cv.imshow('Sobel_x',sobelx)
cv.imshow('Sobel_y',sobely)
cv.imwrite('Machine vision\week3\pics/sobelx.jpg',sobelx)
cv.imwrite('Machine vision\week3\pics/sobely.jpg',sobely)
combined_sobel = cv.bitwise_or(sobelx,sobely)
cv.imshow('combined_sobel',combined_sobel)
cv.imwrite('Machine vision\week3\pics/combined_sobel.jpg',combined_sobel)

# Canny
canny = cv.Canny(gray,125,175)
cv.imshow('canny',canny)
cv.imwrite('Machine vision\week3\pics/canny.jpg',canny)

cv.waitKey(0)