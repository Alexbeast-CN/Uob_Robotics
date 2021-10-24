import cv2 as cv

img = cv.imread('Machine vision\week2\pics/1.jpg')
cv.imshow('org',img)

# 1. Converting to grayscale
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#cv.imshow('gray',gray)

# 2. Blur
# ksize for blue has to be odd number (1,3,5..)
blur = cv.GaussianBlur(img,(3,3),cv.BORDER_DEFAULT)
#cv.imshow('Blur',blur)

# 3. Edge Cascade
canny = cv.Canny(img,125,175)
#cv.imshow('Cannt',canny)

# 4. 扩张 the image
dilated = cv.dilate(canny, (3,3), iterations=3)
#cv.imshow('dilated',dilated)

# 5. 侵蚀
eroded = cv.erode(dilated,(3,3),iterations=1)
#cv.imshow('eroded',eroded)

# 6. resize
resized = cv.resize(img,(500,500),interpolation=cv.INTER_CUBIC)
#cv.imshow('resize',resized)

# 7. Cropping
cropped = img[30:500,100:400]
cv.imshow('cropped',cropped)

cv.waitKey(0)