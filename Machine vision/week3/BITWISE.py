import cv2 as cv
import numpy as np

blank = np.zeros((400,400),dtype='uint8')

rectangle = cv.rectangle(blank.copy(),(30,30),(370,370),255, -1)
circle = cv.circle(blank.copy(),(200,200),200,255,-1)

cv.imshow('rectangle',rectangle)
cv.imshow('circle',circle)
cv.imwrite('Machine vision\week3\pics/rectangle.jpg',rectangle)
cv.imwrite('Machine vision\week3\pics/circle.jpg',circle)

# bitwise AND --> Intersecting regins
bitwise_and = cv.bitwise_and(rectangle,circle)
cv.imshow('bitwise_and',bitwise_and)
cv.imwrite('Machine vision\week3\pics/bitwise_and.jpg',bitwise_and)

# bitwise OR --> Non-Intersecting regins and Intersecting regins
bitwise_or = cv.bitwise_or(rectangle,circle)
cv.imshow('bitwise_or',bitwise_or)
cv.imwrite('Machine vision\week3\pics/bitwise_or.jpg',bitwise_or)

# bitwise XOR --> Non-Intersecting regins
bitwise_xor = cv.bitwise_xor(rectangle,circle)
cv.imshow('bitwise_xor',bitwise_xor)
cv.imwrite('Machine vision\week3\pics/bitwise_xor.jpg',bitwise_xor)

# bitwise NOT --> Invert black and white
bitwise_not = cv.bitwise_not(rectangle)
cv.imshow('bitwise_not',bitwise_not)
cv.imwrite('Machine vision\week3\pics/bitwise_not.jpg',bitwise_not)

cv.waitKey(0)