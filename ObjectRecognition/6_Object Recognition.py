# importing the modules
import cv2 
from PIL import Image,ImageFile
import socket
import time
import io
import requests
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True
CHUNK_LENGTH = 200
#CHUNK_LENGTH = 1460

ipESP32 = '192.168.43.211'
ipPC = '192.168.43.96'

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
s.bind((ipPC, 8000))


fps = 0
geserImage = 50

img_bytes = b''
lenData = 0


# setting some variables
thres = 0.55
nmsThres = 0.2
cap = cv2.VideoCapture(0) 
cap.set(3, 640)
cap.set(4, 480)

classNames = []
classFile = 'Files/coco.names'

# loading the models
with open(classFile, 'rt') as f:
    classNames = f.read().split('\n')

# configuring the paths
configPath = 'Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = "Files/frozen_inference_graph.pb"

# setting the detection model parameters
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

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
            


            '''
            # reading the image frame by frame
            success, img = cap.read()
            # resizing the frame to 640 by 480 
            img = cv2.resize(img,(640,480))
            # detecting the object in captured image
            '''
            classIds, confs, bbox = net.detect(frame, confThreshold=thres, nmsThreshold=nmsThres)
            try:
                send = ''
                # looping throught the detected objects
                for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    # drawing the rectangle around the dected objects
                    cv2.rectangle(frame, (box[0],box[1]), (box[2],box[3]), (255,0,0), 1)
                    # drawing the detected object names
                    cv2.putText(frame, f'{classNames[classId - 1].upper()} {round(conf * 100, 2)}',
                                (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1, (0, 255, 0), 1)
                    if send == '':
                        send = classNames[classId - 1].upper()
                    else:
                        send = send +','+classNames[classId - 1].upper()
                if send != '':
                    requests.get('http://'+ipESP32+':8080/?data='+send)
            except:
                print('...')






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

