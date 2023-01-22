import os
import cv2 as cv
import numpy as np

def rescaleFrame(frame, scale=0.30): # This works for images, videos and live video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width,height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

people = ['Bibek Ray', 'Elizabeth Olsen', 'Robert Downey jr', 'Tom Holland']
DIR = r'C:\Users\Bibek Ray\OneDrive\Desktop\Live Face Recognition\train'

haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []
def create_train():

    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            print(img_path)
            img_arr = cv.imread(img_path)
            #img_arr = rescaleFrame(img_array)
            #cv.imshow('', img_arr)
            #cv.waitKey(1)
            if img_arr is not None:
                gray = cv.cvtColor(img_arr, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()

print('Training completed')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the recognizer on the features list and the labels list
face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)