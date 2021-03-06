---
layout: post
title: Face Detection
---
![2019-4-23-Face-Detection](/images/face_detection_output.png "2019-4-23-Face-Detection")

As it can be seen above, the model is able to detect all of the faces in the image correctly. *GoT Spoiler Alert* I believe the detector will fail at Arya Stark’s face; but we shouldn’t be blaming it on the detector itself as she is a faceless now.

Joking aside, this detector is fast thanks to the cascades being fast at computation; however, it works better mostly at frontal images and either gives users false-positives easily or misses the presence of a face.

Now that there’s HOG and CNNs at our disposal to build detectors, it can be said that this is not the best face detector; but does well enough using only OpenCV library. Besides, improvements to some extend can be achieved although the parameters are not easy to tune. 


**Python Code Block:**

```python

import cv2
import argparse

# argument parsers
ap = argparse.ArgumentParser()
ap.add_argument("--cascade", required=True, help="Path to where the face cascade is") #pre-trained face detector provided by OpenCV
ap.add_argument("--image", required=True, help="Path to where the image is")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# load the face detector and detect faces in the image
detector = cv2.CascadeClassifier(args["cascade"])

faces = detector.detectMultiScale(gray, scaleFactor=1.07, minNeighbors=5,
                                  minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

# loop over the faces and draw a bounding box around each of them
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

print("{} face(s) found".format(len(faces)))

# display the detected faces
cv2.imshow("Face Detection", image)
cv2.waitKey(0)

#Parameters of cv2.CascadeClassifier.detectMultiScale():
#cascade – Haar classifier cascade (OpenCV 1.x API only). It can be loaded from XML or YAML file using Load(). When the cascade is not needed anymore, release it using cvReleaseHaarClassifierCascade(&cascade).
#image – Matrix of the type CV_8U containing an image where objects are detected.
#objects – Vector of rectangles where each rectangle contains the detected object.
#scaleFactor – Parameter specifying how much the image size is reduced at each image scale.
#minNeighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it.
#flags – Parameter with the same meaning for an old cascade as in the function cvHaarDetectObjects. It is not used for a new cascade.
#minSize – Minimum possible object size. Objects smaller than that are ignored.
#maxSize – Maximum possible object size. Objects larger than that are ignored.

#as a debugging rule; start with the scaleFactor, adjust it as needed, and then move on to minNeighbors

#capable of running in real-time; but prone to False Positives and parameters can be hard to tune

```
*The image was taken from <https://ksassets.timeincuk.net/wp/uploads/sites/55/2017/08/2017_GameOfThrones_HBO_220817-920x584.jpg>
