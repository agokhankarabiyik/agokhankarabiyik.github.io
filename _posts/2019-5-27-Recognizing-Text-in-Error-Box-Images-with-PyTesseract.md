---
layout: post
title: Recognizing Text in Error Box Images with PyTesseract
---
![2019-5-27-Recognizing-Text-in-Error-Box-Images-with-PyTesseract](/images/error1.png "2019-5-27-Recognizing-Text-in-Error-Box-Images-with-PyTesseract")

All of us face many errors throughout the day

What I aim here in this post is to be able to contribute to the automated IT Desk solutions. 

Imagine one of your customers is using your software and they've just had an error and it's been displayed into their screen. 

Steps 


**Python Code Block:**

First of all, I imported necessary packages and added argument parser for easy use of command line; then, I assigned a for-loop to go through my dataset and applied some basic computer vision operations such as resizing, converting the images of interest to the grayscale, blurring and thresholding them. 

```python

import argparse
import cv2
import numpy as np
import imutils
from imutils import paths
from PIL import Image
import pytesseract
import os


ap = argparse.ArgumentParser()
ap.add_argument("--images", required = True, help = "path to the image folder")
args = vars(ap.parse_args())

for imagePath in sorted(list(paths.list_images(args["images"]))):
    print(imagePath[imagePath.rfind("/") + 1:])
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width = 1024)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    (T, thresh) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    cv2.imshow("Thresholded", thresh)
    print("Otsu's T value: {}".format(T))
    #cv2.waitKey(0)
    
```
   
The first block of code was straight-forward and easy to run as these operations are kind of 'must-do' before proceeding further in almost any sort of computer vision related task. 

The output of one of the images after thresholding can be seen below.

![2019-5-27-Recognizing-Text-in-Error-Box-Images-with-PyTesseract](/images/error1_thresholded.png "2019-5-27-Recognizing-Text-in-Error-Box-Images-with-PyTesseract")  

The real deal has just started! What I was planning to do was crop the error box from image using an edge detector and feed that into Tesseract. Needless to say, it didn't work the way I expected. Let me tell you why and what steps I followed to tackle those problems.

As mentioned above, the images that I put into my dataset folder were random images that I found on Google Images with error boxes in them. Therefore, all of them were different kind of errors and there was no consistency in both the length of the text in the box and the shape of the box. 



```python

    for i in range(0, 6):
        dilated = cv2.dilate(thresh.copy(), None, iterations = i + 1)
    cv2.imshow("Dilated {} times".format(i + 1), dilated)
    #cv2.waitKey(0)

    clone = dilated.copy()
    _, cnts, _ = cv2.findContours(clone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

```
![2019-5-27-Recognizing-Text-in-Error-Box-Images-with-PyTesseract](/images/error1_dilated.png "2019-5-27-Recognizing-Text-in-Error-Box-Images-with-PyTesseract")

```python

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("boxed", image)

        roi = image[y:y+h, x:x+w]

        cv2.imshow('segment no:' + str(i), roi)
        #cv2.waitKey(0)
        
```
![2019-5-27-Recognizing-Text-in-Error-Box-Images-with-PyTesseract](/images/error1_contour_segments.png "2019-5-27-Recognizing-Text-in-Error-Box-Images-with-PyTesseract")


```python

        filename = "{}.png".format(os.getpid())
        cv2.imwrite(filename, image)

        text = pytesseract.image_to_string(Image.open(filename))
        os.remove(filename)
        print(text)

```
![2019-5-27-Recognizing-Text-in-Error-Box-Images-with-PyTesseract](/images/error1_tesseract_output.png "2019-5-27-Recognizing-Text-in-Error-Box-Images-with-PyTesseract")


```python


        file = open((imagePath.rsplit(".", 1)[0]) + ".txt", "a+")
        file.write(text)
        file.close()

```
![2019-5-27-Recognizing-Text-in-Error-Box-Images-with-PyTesseract](/images/error1_text_file.png "2019-5-27-Recognizing-Text-in-Error-Box-Images-with-PyTesseract")

```python

print("[INFO]: ERROR TYPES")

for imagePath in sorted(list(paths.list_images(args["images"]))):
    print(imagePath[imagePath.rfind("/") + 1:] + ':')
    
    if ('server' and 'cannot') in open(imagePath.rsplit(".", 1)[0] + '.txt').read():
        print("error1")
    elif ('unable' and 'login') in open(imagePath.rsplit(".", 1)[0] + '.txt').read():
        print("error2")
    elif ('replace user' and 'serious') in open(imagePath.rsplit(".", 1)[0] + '.txt').read():
        print('error6')
    elif ('module' and 'not' and 'found') in open(imagePath.rsplit(".", 1)[0] + '.txt').read():
        print("error8")
    else:
        print("none")
        
```
![2019-5-27-Recognizing-Text-in-Error-Box-Images-with-PyTesseract](/images/error_types_output.png "2019-5-27-Recognizing-Text-in-Error-Box-Images-with-PyTesseract")

```python

search_word = input("enter the word(s) you want to search in file(s): ")
for imagePath in sorted(list(paths.list_images(args["images"]))):
    file = open(imagePath.rsplit(".", 1)[0] + '.txt')
    strings = file.read()

    if(search_word in strings):
        print("word found in " + imagePath[imagePath.rfind("/") + 1:])


```
![2019-5-27-Recognizing-Text-in-Error-Box-Images-with-PyTesseract](/images/error_search.png "2019-5-27-Recognizing-Text-in-Error-Box-Images-with-PyTesseract")




![2019-5-27-Recognizing-Text-in-Error-Box-Images-with-PyTesseract](/images/error10_bounding_rectangles.png "2019-5-27-Recognizing-Text-in-Error-Box-Images-with-PyTesseract")

![2019-5-27-Recognizing-Text-in-Error-Box-Images-with-PyTesseract](/images/error10_text_file.png "2019-5-27-Recognizing-Text-in-Error-Box-Images-with-PyTesseract")


*The images were taken from: 
Error1: <https://pjrjx47372.i.lithium.com/t5/image/serverpage/image-id/9002i069ABAE234BCD2D4/image-size/large?v=1.0&px=999>
Error10: <https://communities.bentley.com/cfs-file/__key/communityserver-discussions-components-files/283906/gINT-screen.jpg>
