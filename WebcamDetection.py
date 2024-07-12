import cv2
import matplotlib.pyplot as plt

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 0 for default camera. If we have webcams, we can use different numbers.
webcam_access = cv2.VideoCapture(0)

# This is similar to the classifier function, this takes an image and activates the classifier on it, drawing it on CV2.

def face_detection(camera):
    grayScaleConverter = cv2.cvtColor(camera, cv2.COLOR_BGR2GRAY)
    detection = face_classifier.detectMultiScale(grayScaleConverter, 1.1, 5, minSize=(100, 100))
    for (x, y, w, h) in detection:
        cv2.rectangle(camera, (x, y), (x+w, y+h), (0, 200, 200), 4)
    #colorConverter = cv2.cvtColor(camera, cv2.COLOR_GRAY2RGB)
    #return colorConverter

detection_activation = True

while detection_activation:
    res, frame = webcam_access.read() # SImilar to imread.
    if res is False:
        break # Stop the sequence if there is a

    face_detection(frame)

    cv2.imshow("Face Detection", frame)

    # cv2.waitkey display the frames between miliseconds. Waitkey(1) means each frame is 1ms apart from each other - meaning almost continous. THe higher the parameter, the lower the frame rate.
    if (cv2.waitKey(1) & 0xFF == ord("q")):
        break

webcam_access.release()
cv2.destroyAllWindows()

    

