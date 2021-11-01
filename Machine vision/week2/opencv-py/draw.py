import numpy as np
import cv2 as cv

blank = np.zeros((500,500,3),dtype='uint8')
#cv.imshow('blank', blank)

# 1. paint the image a certain color
# Remember different from we normally use rgb images
# in opencv the color sequence is bgr
blank[200:300,300:400] = 203,192,255
#cv.imshow('pink',blank)

# 2. Draw a Rectangle
# 由长方形的两个角画出这个长方形
#cv.rectangle(blank,(0,0),(250,250),(203,192,255),thickness=2)
#cv.imshow('Rectangle',blank)

# 当然也可以利用原图片的大小来画我们的长方形
cv.rectangle(blank,(0,0),(blank.shape[1]//2,blank.shape[0]//2),(203,192,255),thickness=-1)
#cv.imshow('Rectangle',blank)


# 3. Draw a circle
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2),40,(173,216,230),thickness=3)
#cv.imshow('Circle',blank)

# 4. Draw a line
cv.line(blank, (0,0) ,(blank.shape[1]//2, blank.shape[0]//2),(255,255,255),thickness=3)
#cv.imshow('line',blank)

# 5. Write text
cv.putText(blank,'Hello World',(250,400),cv.FONT_HERSHEY_TRIPLEX, 1.0, (255,255,255),2)
cv.imshow('text',blank)

cv.imshow()
cv.waitKey(0)

