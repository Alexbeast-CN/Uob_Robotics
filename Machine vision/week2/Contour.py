import cv2 as cv
import numpy as np

img = cv.imread('Machine vision\week2\pics\cats.jpg')
cv.imshow('cats',img)

blank = np.zeros(img.shape[:2], dtype='uint8')

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# blur = cv.blur(gray,(3,3),cv.BORDER_DEFAULT)
# canny = cv.Canny(blur,125, 175)
# cv.imshow('canny',canny)

ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow('thresh',thresh)

contours, hierarchies = cv.findContours(thresh,cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) 

# To draw all the contours on the blank pic.
# use -1 to represente all
cv.drawContours(blank, contours, -1,(255,255,255),thickness=2)
cv.imshow('Contour Drawn', blank)

cv.waitKey(0)