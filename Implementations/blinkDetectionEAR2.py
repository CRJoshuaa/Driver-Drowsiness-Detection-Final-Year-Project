from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import pandas as pd
import argparse
import imutils
from datetime import datetime
import time
import dlib
import cv2


def EAR(eye):
    #calculate euclidean distance between the eyes (vertical)
    A=dist.euclidean(eye[1],eye[5])
    B=dist.euclidean(eye[2],eye[4])

    #calculate euclidean distance between the eyes (horizontal)
    C=dist.euclidean(eye[0],eye[3])

    #compute EAR
    ear=(A+B)/(2.0 * C)

    return ear

#argument parsing
ap=argparse.ArgumentParser()
ap.add_argument('-v', "--video",type=str,default="",
                help="path to input file")
args=vars(ap.parse_args())

#initalising constants and counts
filename="EAR_lower"
EAR_THRESH=0.23
EAR_CONSEC_FRAMES=2

frame_count=0
num_blinks=0

data=[]


#initialise face detector
print("[UPDATE] Loading facial landmark predictor....")
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#slice the index for the left and right eye
(l_start,l_end)=face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(r_start,r_end)=face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[UPDATE] Starting camera....")
vs=cv2.VideoCapture(args["video"])
fileStream=True
# vs=cv2.VideoCapture(0) ## comment this and next line if using file source
# fileStream=False

frame_width = int(vs.get(3)) 
frame_height = int(vs.get(4)) 
   
size = (frame_width, frame_height) 
print(size)

outputVideo = cv2.VideoWriter(filename+".avi",cv2.VideoWriter_fourcc(*'XVID'), 3, size) 

while (vs.isOpened()):
    isBlink=False

    ret,frame=vs.read()
    #frame=imutils.resize(frame,width=450)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #detects face in the grayscale image
    rects=detector(gray,0)

    for rect in rects:
        #determine the landmarks and convert to numpy array
        shape=predictor(gray,rect)
        shape=face_utils.shape_to_np(shape)

        #extract left and right eye coordinates
        left_eye=shape[l_start:l_end]
        right_eye=shape[r_start:r_end]

        left_EAR=EAR(left_eye)
        right_EAR=EAR(right_eye)

        mean_EAR=(left_EAR+right_EAR)/2.0

        #visualise outlines for the eyes
        left_eye_hull=cv2.convexHull(left_eye)
        right_eye_hull=cv2.convexHull(right_eye)

        cv2.drawContours(frame,[left_eye_hull],-1,(0,255,0),1)
        cv2.drawContours(frame,[right_eye_hull],-1,(0,255,0),1)

        #blink Detection
        if mean_EAR<EAR_THRESH:
            frame_count+=1

        else:
            if frame_count>=EAR_CONSEC_FRAMES:
                num_blinks+=1
                isBlink=True

                frame_count=0

        cv2.putText(frame,"Blinks:{}".format(num_blinks),(10,30),
            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    
        #adding data to database


        data.append([mean_EAR,isBlink])
        isBlink=False

 
    
    outputVideo.write(frame)

    #display frame
    cv2.imshow("Frame",frame)
    #time.sleep(0.1)
    key=cv2.waitKey(1) & 0xFF

    if key==ord("q"):
        df = pd.DataFrame(data, columns = ['EAR', 'Blink'])
        filename="blinkData_"+datetime.now().strftime("%Y-%m-%d %H-%M")+".csv"
        df.to_csv(filename)

        print(frame.shape)

        break
    
outputVideo.release()
vs.release()
cv2.destroyAllWindows()

