---
layout: post
title: Recognizing Text in Error Box Images with PyTesseract
---
![2019-5-27-Recognizing-Text-in-Error-Box-Images-with-PyTesseract](/images/error1.png "2019-5-27-Recognizing-Text-in-Error-Box-Images-with-PyTesseract")

What I aim here in this post is to be able to get images or screenshots containing error boxes and feed them into PyTesseract after some operations in OpenCV to recognize text so that I can classify them by the type of error and take actions accordingly. Thus, the workload of IT Help Desks could be reduced after this automation.

**Python Code Block:**

First of all, I imported necessary packages and added argument parser for easy use of command line; then, I assigned a for-loop to go through my dataset and applied some basic computer vision operations such as resizing, converting the images of interest to the grayscale, blurring and thresholding them before proceeding further which is quite a common practice in almost every single computer vision application.

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
   
![2019-5-27-Recognizing-Text-in-Error-Box-Images-with-PyTesseract](/images/error1_thresholded.png "2019-5-27-Recognizing-Text-in-Error-Box-Images-with-PyTesseract")  

The real deal has just started! What I was planning to do was crop the error box from image using an edge detector and feed that into Tesseract. Needless to say, it didn't work the way I expected. Let me tell you why and what steps I followed to tackle those problems from my logbook.

The images that I put into my dataset folder were random images that I found on Google Images with error boxes in them. Therefore, all of them were different kind of errors and there was no consistency in both the length of the text in the box and the shape of the box. 

The first thing I did was to apply auto canny edge detector and I was hoping to crop the error box. I had a good output; but, some of the edges were discontinued after applying the detector although the canny edge detector is well-known for this job. I also tried to find contours in the image and sort them to find a parent contour with many child contours (RECT_TREE); but, somehow, it didn't work well again. That's why I wanted to try to convert the image to binary. I applied blackhat morphological operation which ended up fine. The text was perfectly visible to human eye, it still wasn't easy to crop for computer though.

HOG + Linear SVM was another option to localize the ROI before extracting; but I didn't try it as HOG doesn’t work well with different aspect ratios. And I wasn't sure if it'd be worth the time I would spend if I created a training dataset and train the model etc.

Then, I changed my approach and decided to apply dilation to images a few times to obtain white blobs in place of the text. I found the contours in the images and used the bounding boxes of these contours to temporarily segment the images then fed them into PyTesseract. It is, apparently, not the best solution; however, it worked well enough under these circumstances.

```python

    for i in range(0, 6):
        dilated = cv2.dilate(thresh.copy(), None, iterations = i + 1)
    cv2.imshow("Dilated {} times".format(i + 1), dilated)
    #cv2.waitKey(0)

    clone = dilated.copy()
    _, cnts, _ = cv2.findContours(clone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

```

![2019-5-27-Recognizing-Text-in-Error-Box-Images-with-PyTesseract](/images/error1_dilated.png "2019-5-27-Recognizing-Text-in-Error-Box-Images-with-PyTesseract")

After the dilation process, I took advantage of the bounding boxes of contours to segment the blobs in the image. Of course, my error text was not the only blob with or without meaningful word chunks in the entire image. There were some other parts of the image which was segmented after dilation to send through PyTesseract.

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

Below, I saved the segmentd image on my disc temporarily just to be able to feed every single segmented image into PyTesseract, then I deleted the saved image.

```python

        filename = "{}.png".format(os.getpid())
        cv2.imwrite(filename, image)

        text = pytesseract.image_to_string(Image.open(filename))
        os.remove(filename)
        print(text)

```
![2019-5-27-Recognizing-Text-in-Error-Box-Images-with-PyTesseract](/images/error1_tesseract_output.png "2019-5-27-Recognizing-Text-in-Error-Box-Images-with-PyTesseract")

Next, I opened a new text file and saved the output of PyTesseract into that file .

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
