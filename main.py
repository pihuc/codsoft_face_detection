import cv2
print (cv2.__version__)

import numpy as np
#face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def my_face_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3, 5)
    if faces is ():
        return img

    #for (x,y,w,h) in faces:
    #    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    for (x, y, w, h) in faces:
        center_coordinates = x + w // 2, y + h // 2
        radius = w // 2
        cv2.circle(img, center_coordinates, radius, (57, 255, 20), 5)
    return img



cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = my_face_detection(frame)

    cv2.imshow('Video Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

