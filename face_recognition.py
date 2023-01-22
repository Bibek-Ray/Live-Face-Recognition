from cv2 import COLOR_BGR2GRAY, FONT_HERSHEY_COMPLEX, cvtColor
import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people =  ['Bibek Ray', 'Elizabeth Olsen', 'Robert Downey jr','Tom Holland']

#feature = np.load('features.npy')
#labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

capture = cv.VideoCapture(0)
"""
img = cv.imread(r'Filepath')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

#Detect the faces in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+h]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'label = {people[label]} with confidence of {confidence}')

    cv.putText(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness = 2)
    cv.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), thickness = 2)

cv.imshow('Detected Face', img)
"""
while True:
    isTrue, frame = capture.read()
    frame_resize = frame
    gray = cv.cvtColor(frame_resize, cv.COLOR_BGR2GRAY)
    #cv.imshow('Person', gray)
    # Detect the face in the imaged
    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)
    for (x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(faces_roi)
        value = people[label] + "; Confidence =" + str(confidence)
        cv.rectangle(frame_resize, (x,y), (x+w, y+h), (0, 255, 0), thickness = 2)
        cv.putText(frame_resize, str(value), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (255,0,0), thickness = 2)
        cv.imshow('Detected', frame_resize)
    #print(f'label = {people[label]} with a confidence of {confidence}')
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

    

#cv.imshow('DEtected Face', capture)

#cv.waitKey(0)