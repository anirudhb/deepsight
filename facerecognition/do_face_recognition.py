### IMPORTS

import sys
import cv2 as cv

### PREINIT

if len(sys.argv) < 4:
    print("Expected 3 arguments: input, face, labelled")

### INIT

face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv.VideoCapture(0)

### RUN

while True:
    ret, img = cap.read()
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey, 1.3, 5)
    for x, y, w, h in faces:
        fs = 0.5
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        tw, th = cv.getTextSize("Face", cv.FONT_HERSHEY_DUPLEX, fs, 1)[0]
        cv.rectangle(img, (x, y), (x + tw, y - th), (255, 0, 0), cv.FILLED)
        cv.putText(img, "Face", (x, y), cv.FONT_HERSHEY_DUPLEX, fs, (0, 0, 255), 1)
    cv.imshow("img", img)
    cv.waitKey(1)

cap.release()
cv.destroyAllWindows()