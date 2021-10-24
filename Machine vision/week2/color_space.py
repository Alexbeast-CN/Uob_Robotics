import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('Machine vision\week2\pics\cats.jpg')
cv.imshow('cats',img)

# what does a BGR image looks like in RGB 
# plt.imshow(img)
# plt.show()

# 1. BGR to GRAY
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow('gray',gray)

# 2. BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# cv.imshow('hsv',hsv)

# 3. BRG to L*A*B
LAB = cv.cvtColor(img, cv.COLOR_BGR2LAB)
# cv.imshow('LAB',LAB)

# 4. BGR to RGB
RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# cv.imshow('RGB',RGB)

plt.imshow(RGB)
# plt.show()

cv.waitKey(0)