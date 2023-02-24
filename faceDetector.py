import cv2
from random import *

#load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_faced_data = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

#getting webcam (u can add any vieo instead of 0)
webCam = cv2.VideoCapture(0)

#iterate forever over frames
while True:
    #Read the current frame
    successful_frame_read, frame = webCam.read()
    #grayscaled_image
    grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detect faces
    face_coordinates = trained_faced_data.detectMultiScale(grey_image)

    #draw a rectangle around the image
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),2)
    #display the image with rectangle around the face
    cv2.imshow('face',frame)
    key = cv2.waitKey(1)
    if key==81 or key==113:
        break

"""
#choose an image to detect faces in 
img = cv2.imread('faces.jpg')


#grayscaled_image
grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect faces
face_coordinates = trained_faced_data.detectMultiScale(grey_image)

#draw a rectangle around the image
for (x,y,w,h) in face_coordinates:
    print(x)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

#display the image with rectangle around the face
cv2.imshow('face',img)
cv2.waitKey()

"""

print("Code Completed")