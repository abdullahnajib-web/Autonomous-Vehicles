from os.path import dirname
import face_recognition
import cv2
import os
import time
import numpy as np
from PIL import Image,ImageFile
import socket
import io
import requests


ImageFile.LOAD_TRUNCATED_IMAGES = True
CHUNK_LENGTH = 200
#CHUNK_LENGTH = 1460

ipESP32 = '192.168.43.211'
ipPC = '192.168.43.96'

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
s.bind((ipPC, 8000))


def findEncodings(images):
    encodeList =[]
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

current_dir = dirname(__file__)

images=[]
names=[]

imagesList=os.listdir(current_dir+"\\train")
for cl in imagesList:
    curImg = cv2.imread(f'{current_dir}\\train\\{cl}')
    images.append(curImg)
    names.append(os.path.splitext(cl)[0])

encodeList = findEncodings(images)

fps = 0
geserImage = 50

img_bytes = b''
lenData = 0
while True:

    data, IP = s.recvfrom(10000)
    data_size = len(data)
    if data_size == CHUNK_LENGTH and data[0] == 255 and data[1] == 216 and data[2] == 255 : # FF D8 FF        
        img_bytes = b''

    img_bytes = img_bytes + data
    lenData = lenData + data_size
    
    if data_size != CHUNK_LENGTH and data[data_size - 2] == 255 and data[data_size - 1] == 217 : # FF D9
        lenKirim = int.from_bytes(bytes([data[data_size - 4],data[data_size - 3]]), "big")
        #print(lenKirim,' ',lenData)
        if lenKirim == lenData:
            start = time.time()
            byteImgIO = io.BytesIO()
            byteImgIO.write(img_bytes)
            image = Image.open(byteImgIO)
            frame = cv2.cvtColor(np.asarray(image),cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame,1)
            
            encode = face_recognition.face_encodings(frame)
            if encode:
                faceLoc = face_recognition.face_locations(frame)
                faceDis = face_recognition.face_distance(encodeList,encode[0])
                indeks = np.argmin(faceDis)
                name = names[indeks].upper()
                #print(faceLoc[0])
                y1,x2,y2,x1 = faceLoc[0]
                x1,y1,x2,y2= x1-geserImage,y1-geserImage,x2+geserImage,y2+geserImage
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)        
                cv2.rectangle(frame,(x1,y2-35),(x2,y2),(255,0,0),cv2.FILLED)        
                if faceDis[indeks] < 0.5:
                    cv2.putText(frame, name, (x1+6,y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                else:
                    name = "UNKNOWN"
                    cv2.putText(frame, "UNKNOWN", (x1+6,y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                print(faceDis)
                #send name
                #requests.get('http://'+ipESP32+':8080/?data='+name)
            end = time.time()
            totalTime = end - start
            if totalTime != 0 :
                fps = 1 / totalTime
            cv2.putText(frame, f'FPS: {int(fps)}', (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,255), 2)
            cv2.imshow("Image", frame)
        
        lenData = 0
        if cv2.waitKey(1) == ord('q'):
            break
cv2.destroyAllWindows()






