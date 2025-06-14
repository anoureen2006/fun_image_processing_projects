import cv2 
import numpy as np
import time

# Load the face and eye classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Capture video from webcam
cap = cv2.VideoCapture(0)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at ({x}, {y})")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5,minSize=(280, 280))
  
    if len(faces) > 0:
       # print('focused')
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            blurredd=cv2.GaussianBlur(roi_gray,(7,7),0)
            threshh= cv2.adaptiveThreshold(blurredd, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            contourss,_=cv2.findContours(threshh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            contourss=sorted(contourss,key=lambda x:cv2.contourArea(x),reverse=True)
            
            cv2.imshow('jjjjj',threshh)
           
            
            # Detect eyes within the face region
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(40, 40))
          
            
            for (ex, ey, ew, eh) in eyes:
                eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
                eye_color = roi_color[ey:ey+eh, ex:ex+ew]
                blurred=cv2.GaussianBlur(eye_gray,(7,7),0)
                thresh= cv2.adaptiveThreshold(eye_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                contours,_=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                contours=sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)

                cv2.imshow('contours',thresh)
                if contours:
                    (cx, cy), radius = cv2.minEnclosingCircle(contours[0])
                    cv2.circle(eye_color, (int(cx), int(cy)), int(radius), (255, 0, 0), 2)
                   
                    
                    if (cx < 643.0) & (cx > 1251.0):
                      print("distracted")
                    if cy >650:
                      print("distracted")
                else:
                    print(" distracted")
    else:
          print('not focused')
            

    cv2.setMouseCallback('Eye Tracking', mouse_callback)
    cv2.imshow('Eye Tracking', frame)
  
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()