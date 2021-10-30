import cv2 as cv

img = cv.imread('Machine vision\week4\pics/Black_pink.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#cv.imshow('Lisa_gray',gray)

haar_cascade = cv.CascadeClassifier('Machine vision\week4/harr_face.xml')

# By changing the value of minNeighbor according to the face pixel area,
# we can get a better result. But cascade is sensetive to noise, it's not
# an advanced method but a popular one.
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.05,minNeighbors=6,minSize=[80,80])

print(f'Number of faces found = {len(faces_rect)}')

for(x,y,w,h) in faces_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow('Face_detect',img)
cv.imwrite('Machine vision\week3\pics/Black_pink_detect.jpg',img)

cv.waitKey(0)