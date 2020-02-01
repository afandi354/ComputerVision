import numpy as np
import cv2 as cv

wajah = cv.CascadeClassifier(r'model\haarcascade_frontalface_default.xml')
mata = cv.CascadeClassifier(r'model\haarcascade_eye.xml')

img = cv.imread('hash.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

deteksi_wajah = wajah.detectMultiScale(img_gray, 1.3, 5)
for(x,y,w,h) in deteksi_wajah:
    cv.rectangle(img,(x,y), (x+w,y+h),(255,0,0),2)
    roi_gray=img_gray[y:y+h, x:x+w]
    roi_color=img[y:y+h, x:x+w]
    deteksi_mata=mata.detectMultiScale(roi_gray)
for(ex, ey, ew, eh) in deteksi_wajah:
    cv.rectangle(roi_color, (ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv.namedWindow('img', cv.WINDOW_NORMAL)
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()
