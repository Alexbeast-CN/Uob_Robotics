import cv2 as cv

img = cv.imread('Machine vision\week2\pics/1.jpg')
cv.imshow('org',img)

# average 
# 此方法的原理就是将一个方形矩阵中间格周边的格子求平均后放在中心的像素上。
average = cv.blur(img,(3,3))
cv.imshow('average blur', average)
cv.imwrite('Machine vision\week3\pics/average_blur.jpg',average)
# Gaussian Blur
# 与 average 唯一不同的是高斯模糊使用了加权平均
Gaussian = cv.GaussianBlur(img,(3,3),0)
cv.imshow('Gaussian blur', Gaussian)
cv.imwrite('Machine vision\week3\pics/Gaussian_blur.jpg',Gaussian)

# Median Blur
# Median 直接选取了中间像素的值
Median = cv.medianBlur(img,3)
cv.imshow('Median blur', Median)
cv.imwrite('Machine vision\week3\pics/Median_blur.jpg',Median)

# Bilateral 
# Bilateral 的方法与
Bilateral = cv.bilateralFilter(img, 5, 15, 15)
cv.imshow('Bilateral', Bilateral)
cv.imwrite('Machine vision\week3\pics/Bilateral.jpg',Bilateral)


# 模糊程度排名：
# Average > Gaussian > Median > Bilateral

cv.waitKey(0)