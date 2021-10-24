import cv2 as cv
import numpy as np

img = cv.imread('Machine vision\week2\pics/1.jpg')
cv.imshow('org',img)

# 1. Translation
# 将图片上下左右移动

def translate(img, x, y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    demensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, demensions)

# -x --> Left
# -y --> Up
# x --> Right
# y --> Down

translated = translate(img, -100, -100)
#cv.imshow('translate',translated)

# 2. Rotation
# 旋转图片

def rotate(img, angle, rotPoint=None):
    (hight, width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2,hight//2)
    
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimension = (width,hight)

    return cv.warpAffine(img, rotMat, dimension)

rotated = rotate(img,45)
#cv.imshow('rotate',rotated)

# 3. Flipping (反转)

flip1 = cv.flip(img, 1)
flip2 = cv.flip(img, -1)
cv.imshow('flip1',flip1)
cv.imshow('flip2',flip2)


cv.waitKey(0)