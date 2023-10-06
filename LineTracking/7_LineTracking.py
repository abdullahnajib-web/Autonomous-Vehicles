# importing the modules
import cv2
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


# setting up camera
cap = cv2.VideoCapture(0)
cap.set(3,160)
cap.set(4,120)

# setting lower black and upper black values
low_b = np.uint8([5,5,5])
high_b = np.uint8([0,0,0])

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
        if lenKirim == lenData:
            start = time.time()
            byteImgIO = io.BytesIO()
            byteImgIO.write(img_bytes)
            image = Image.open(byteImgIO)
            frame = cv2.cvtColor(np.asarray(image),cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame,1)

            mask = cv2.inRange(frame,high_b,low_b)
            # finding the controus/areas where is the balck color
            contours, hierarchy = cv2.findContours(mask,1,cv2.CHAIN_APPROX_NONE)

            # checking if any contour is found
            if len(contours) > 0:
                # finding its c and y positional values
                c = max(contours,key=cv2.contourArea)
                M = cv2.moments(c)
                if M['m00'] !=0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])

                    # turning left or right
                    if cx >= 120:
                        requests.get('http://'+ipESP32+':8080/?data=turn left')
                        print('turn left')
                        # add your hardware control here to trun left
                    if cx < 120 and cx > 40:
                        requests.get('http://'+ipESP32+':8080/?data=on track')
                        print('On Track')
                        # add your hardware control here to move 
                    if cx <= 40:
                        requests.get('http://'+ipESP32+':8080/?data=turn right')
                        print('turn right')
                        # add your hardware control here to trun right
                    #print(cx)

            
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

