import cv2 as cv
import face_recognition
import numpy as np
import os
from datetime import datetime

path = 'ImageAttendance'
images = []
classNames = []
mylist = os.listdir(path)
print(mylist)
for cl in mylist :
    curImg = cv.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

#encoding 함수
def findEncodings(images) :
    encodeList = []
    for img in images :
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name) :
    with open('Attendance.csv','r+') as f :
        myDataList = f.readlines()
        nameList = []
        for line in myDataList :
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList :
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


encodeListKnown = findEncodings(images)
print('Encoding Complete')

#webcam
cap = cv.VideoCapture(0)

while True :
    success, img = cap.read()
    imgS = cv.resize(img, (0,0),None,0.25,0.25)
    imgS = cv.cvtColor(imgS, cv.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,faceCurFrame) :
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex] :
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv.FILLED)
            cv.putText(img,name,(x1+6,y2-6),cv.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)

    cv.imshow('Webcam',img)
    cv.waitKey(1)

