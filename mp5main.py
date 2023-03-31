import cv2
import numpy as np
import face_recognition as fr
import os
from datetime import datetime

path = 'imagesMP5'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    currImg = cv2.imread(f'{path}/{cl}')
    images.append(currImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findImageEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def trackAttendance(name):
    with open('markAttendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            currDT = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {currDT}')


encodeListKnown = findImageEncodings(images)
print('Completion of Image Encoding.')

cam = cv2.VideoCapture(0)

while True:
    success, img = cam.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    facesCurrFrame = fr.face_locations(imgS)
    encodeCurrFrame = fr.face_encodings(imgS, facesCurrFrame)

    for encodeFace, faceloc in zip(encodeCurrFrame, facesCurrFrame):
        matches = fr.compare_faces(encodeListKnown, encodeFace)
        faceDist = fr.face_distance(encodeListKnown, encodeFace)
        print(faceDist)
        matchIndex = np.argmin(faceDist)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceloc
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 3)
            trackAttendance(name)


    cv2.imshow('MyWebcamera', img)
    cv2.waitKey(1)