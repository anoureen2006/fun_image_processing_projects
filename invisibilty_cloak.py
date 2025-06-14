import cv2 as cv
import numpy as np
import time

def detect_bground(cam, frames=30):
    time.sleep(2)
    print('detecting background')
    backgrounds=[]
    for i in range(frames):
      ret,frame=cam.read()
      if ret:
          backgrounds.append(frame)
          time.sleep(0.1)
          print('detected background')
          return np.median(backgrounds,axis=0).astype(dtype=np.uint8)
      else:
            print('error')
            break
    
def create_cloak(frame, lower_color, upper_color):
    hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    mask=cv.inRange(hsv,lower_color,upper_color)
    mask=cv.morphologyEx(mask,cv.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask=cv.morphologyEx(mask,cv.MORPH_DILATE,np.ones((3,3),np.uint8))
    return mask

def apply_cloak(frame,mask,background):
    mask_inv=cv.bitwise_not(mask)
    fg=cv.bitwise_and(frame,frame,mask=mask_inv)
    bg=cv.bitwise_and(background,background,mask=mask)
    return cv.add(fg,bg)

def main():
    print('starting')
    cam=cv.VideoCapture(0)
    background=detect_bground(cam)
    lower_color=np.array([90,50,50])
    upper_color=np.array([130,255,255])
    while True:
        ret,frame=cam.read()
        cv.imshow('frame',frame)
        mask=create_cloak(frame,lower_color,upper_color)
        result=apply_cloak(frame,mask,background)
        cv.imshow('frame',frame)
        cv.imshow('cloak',result)
        cv.waitKey(1)
    
if __name__=='__main__':
    main()


