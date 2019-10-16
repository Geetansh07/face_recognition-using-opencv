#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2

#make a Cascade classifier object
face_cascade = cv2.CascadeClassifier("C:\\Users\\geetk\\Downloads\\opencv2\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml")

#Reading the image as it is,[add your own image path]
img = cv2.imread("G:\Images\img1.jpg")

#reading the image as gray scale image
gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#search the co-ordinates of the image
faces = face_cascade.detectMultiScale(gray_image,scaleFactor = 1.05,minNeighbors = 5)

#building a rectangle for a face
for x,y,w,h in faces:
    img = cv2.rectangle(img, (x,y) ,(x+w, y+h),(0,255,0),3)

#resizing the image to fit in the window
resize = cv2.resize(img,(800,600))
    
cv2.imshow("Gray",resize)

cv2.waitKey(0)

cv2.destroyAllWindows()

