import cv2
import os
from tensorflow import keras
from keras.models import load_model
import numpy as np
import time
import argparse
from imutils.video import FileVideoStream
from imutils.video import VideoStream

#argument parsing
ap=argparse.ArgumentParser()
ap.add_argument('-v', "--video",type=str,default="",
                help="path to input file")
args=vars(ap.parse_args())

face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')


print("[UPDATE] Loading model....")
model = keras.models.load_model('models/eyeModel_98acc_7loss.h5')


#initialise
filename="CNN_lower"
frame_count=0
num_blinks=0
CONSEC_FRAMES=2
data=[]
rpred=[99]
lpred=[99]

print("[UPDATE] Starting camera....")
vs=cv2.VideoCapture(args["video"])
fileStream=True
# vs=cv2.VideoCapture(0) ## comment this and next line if using file source
# fileStream=False

frame_width = int(vs.get(3)) 
frame_height = int(vs.get(4)) 
   
size = (frame_width, frame_height) 

outputVideo = cv2.VideoWriter(filename+".avi",cv2.VideoWriter_fourcc(*'XVID'), 3, size) 

while(vs.isOpened()):
    isBlink=False
    eyeState=" "


    ret,frame = vs.read()
    height,width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(50,50))
        r_eye= r_eye/255.0
        r_eye=  r_eye.reshape(50,50,1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = model.predict(r_eye)
        if(rpred<0.5):
            lbl='Open'
        else:
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye,(50,50))
        l_eye= l_eye/255.0
        l_eye=l_eye.reshape(50,50,1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict(l_eye)
        if(lpred<0.5):
            lbl='Open'
        else:
            lbl='Closed'
        break

    if(rpred >= 0.5 and lpred >= 0.5):
        eyeState="Closed"
        frame_count+=1
       
    else:
        eyeState="Open"
        if frame_count>=CONSEC_FRAMES:
            num_blinks+=1
            isBlink=True

            frame_count=0
    
    cv2.putText(frame,"Blinks:{}".format(num_blinks),(10,30),
        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    cv2.putText(frame,"Eye:{}".format(eyeState),(300,30),
            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        
    isBlink=False

    outputVideo.write(frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


outputVideo.release()
vs.release()
cv2.destroyAllWindows()