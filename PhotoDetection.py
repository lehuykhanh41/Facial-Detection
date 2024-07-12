# Facial Detection with OpenCV imitation by KL.

import cv2
import matplotlib.pyplot as plt

# Read the image data using CV2
# Sample Images are put inside Data_Image
photoPath = './Data_Image/Face2.jpg' 
photo = cv2.imread(photoPath)

# Then convert the photo to gray scale. THis allows the classifier to works better.
grayscaleFace = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)

# Initialize the Cascade Classifier. THis is a pre-trained model.
# More types can be found here: https://github.com/opencv/opencv/tree/master/data/haarcascades.
face_classifier = cv2.CascadeClassifier (cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initiate the classifier.
detection = face_classifier.detectMultiScale(grayscaleFace, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

# Now that the detection is done, we will draw the bounding box.
# Param lists:
# img: The image needs to be detected.
# x, y: The axis of the image, along with the width and height.
# 0, 255, 0: THis can be used to change the color of the box.
# 4: Thickness.

#-> Obtained from the detectMultiScale.

# COLOR: BGR
for (x, y, w, h) in detection:
    cv2.rectangle(photo, (x, y), (x+w, y+h), (128, 255, 200), 6)

    # (Source, Origin, width & height, box color, box thickess. For full-filled, use -1) 

# Now convert the photo back to color.
rgbFace= cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)

# Finally, plot the face.
plt.figure(figsize=(9.5, 9.5), facecolor="#AAAAAA")

plt.imshow(rgbFace)
plt.axis('off')
plt.title("Phát Hiện Khuôn Mặt")
plt.tight_layout()
plt.show()