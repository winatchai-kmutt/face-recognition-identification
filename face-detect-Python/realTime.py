import cv2
import face_recognition
import numpy as np
import face_recognition
import os
import requests
import json
from datetime import datetime

path = 'imag'
encodeList = []
className = []

def getData() :
    url = 'http://localhost:3000/api/saved'
    data = requests.get(url).json() 
    return data

def fineEndodings(images) :
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        print(encode)
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
fromDB = getData()
for f in fromDB['pofiles'] :
    encode = json.loads(f["encode"])
    name = f["name"]
    encode = np.array(encode)
    encodeList.append(encode)
    className.append(name)
print(encodeList)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faceCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, faceCurFrame) :
        mathes = face_recognition.compare_faces(encodeList, encodeFace) 
        faceDis = face_recognition.face_distance(encodeList, encodeFace)
        matchIndex = np.argmin(faceDis)
        per = int((1-faceDis[matchIndex])*100)
        
        if mathes[matchIndex]:  
            name = className[matchIndex].upper()
            text = f'{name} {per}%'
            print(text)
            y1, x2, y2, x1 = faceLoc
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, text, (x1+6, y2-6), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
            markAttendance(text)
    cv2.imshow('webcam', img)
    cv2.waitKey(1)    
   

        
