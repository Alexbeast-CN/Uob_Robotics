import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('Machine vision/week4/harr_face.xml')

Black_pink = ['Jisoo','Jennie','Lisa','Rose']

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread('Machine vision\week4\pics/Blackpink.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.05,minNeighbors=6,minSize=[80,80])

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {Black_pink[label]} with a confidence of {confidence}')

    cv.putText(img, str(Black_pink[label]), (x+w,y+h), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)
cv.imwrite('Machine vision\week4\pics/Black_pink_rec.jpg',img)
cv.waitKey(0)