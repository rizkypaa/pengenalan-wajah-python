import cv2, time
from PIL import Image
video_capture = cv2.VideoCapture(0)
a = 0
recognizer = cv2.face.LBPHFaceRecognizer_create()
cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
recognizer.read('training/training.xml')
id = 0
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255,0,0)
while True:
  a = a + 1  
  check, frame = video_capture.read()  
  print(check)
  print(frame)
  # Capture frame-by-frame
  ret, frame = video_capture.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
  # Draw a rectangle around the faces
  for (x, y, w, h) in faces:
        cv2.imwrite("DataSet/User."+str(id)+"."+str(a)+".jpg",
        gray[y:y+h,x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        id, conf=recognizer.predict(gray[y:y+h,x:x+w])
        if (id == 1):
           id = "rizky" 
           cv2.putText(frame,str(id),(x+w,y+h),fontFace,fontScale,fontColor)
   # Display the resulting frame
  cv2.imshow('Video', frame)
  #key = cv2.waitKey(1)  
  if (cv2.waitKey(1)==ord('q')):
    break
  print(a)
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()