import cv2,os, time
import numpy as np
from PIL import Image
import pickle
import sqlite3
video_capture = cv2.VideoCapture(0)
a = 0
recognizer = cv2.face.LBPHFaceRecognizer_create()
cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
recognizer.read('training/training.xml')
id=0
fontface=cv2.FONT_HERSHEY_SIMPLEX
fontscale=1
fontcolor=(255,204,0)
path='DataSet'
def getProfile(id):
 conn=sqlite3.connect("C:/aiuas/coba1/wajahorang.db")
 cmd="SELECT * FROM orang WHERE id="+str(id)
 cursor=conn.execute(cmd)
 profile=None
 for row in cursor:
  profile=row
 conn.close()
 return profile
while(True):
 check, frame=video_capture.read();
 gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
 faces=faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    );
 for(x,y,w,h) in faces:
  a=a+1
  cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
  id,conf=recognizer.predict(gray[y:y+h,x:x+w])
  profile=getProfile(id)
  if(profile!=None):
   cv2.putText(frame,str(profile[1]),(x,y+h+30),fontface,fontscale,fontcolor)
   #cv2.putText(frame,str(profile[2]),(x,y+h+60),fontface,fontscale,fontcolor)
   #cv2.putText(frame,str(profile[3]),(x,y+h+90),fontface,fontscale,fontcolor)
   #cv2.putText(frame,str(profile[4]),(x,y+h+100),fontface,fontscale,fontcolor);
 cv2.imshow("Video",frame);
 if(cv2.waitKey(1)==ord('q')):
  break
 print(a)
video_capture.release()
cv2.destroyAllWindows()