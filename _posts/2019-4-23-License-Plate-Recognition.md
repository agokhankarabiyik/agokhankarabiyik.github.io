---
layout: post
title: License Plate Recognition
---
![2019-4-23-License-Plate-Recognition](/images/correct_plate_recognition.png "2019-4-23-License-Plate-Recognition")

License Plate Recognition is one of the most widely-used real-world applications of Computer Vision. That's why I was interested in building a recognizer for license plates in the very first months of my Computer Vision journey. After learning about the basics of computer vision with OpenCV and some image descriptors, I wanted to get involved in the coding part. It was very hard and confusing at the beginning; however, once I got into the codes it all started making sense and I realized how powerful very simple and basic computer vision techniques could possibly be such as gradients, contours, thresholding, bitwise operations and so on.   

I've added two different output images; the first one is at the top of the page which is an image of a correctly recognized license plate and the second one is at the bottom of the page which is not as lucky as the first one. The main reason for this failure lies beneath the dataset used to train the character recognizer. Unfortunately, the dataset was not the real-world representation of characters. As a next step of this project, another dataset of characters extracted from real-world images can be put together and the model can be re-trained. Although this task of creating our own dataset sounds tedious and time-consuming, it is absolutely worth trying as it will give us much higher accuracy. 

Speaking of tedious things, I remember one of my mentors who is the greatest of all times saying that “You will spend 80% of your time to gather, clean and re-arrange your data and the remaining 20% will be spent on complaining about how tedious that 80% was.” Apologies if you are reading here and this is not exactly what you said; but the gist was something like this, wasn't it? :)

_The codes below should be seen as a self-study with side notes in the light of the course material provided by Dr. Adrian Rosebrock._

**Python Code Block:**

I used to write code in Jupyter notebook during my Data Science times. However, ever since I started studying Computer Vision I've opted for terminal and started executing scripts by the command line arguments which I find it quite easy and effective now. Additionally, I stopped writing hundreds of lines of code in a single script and started dividing them into chunks and calling them into the main script as an import from my local instead which makes everything tidier and eye-pleasing.

The first block of codes below is simply used to detect license plates in images.

```python

#Dataset: MediaLab National Technical University Athens in Greece
#Assumptions:
#The license plate text is always darker than the license plate background.
#The license plate itself is approximately rectangle.
#The license plate region is wider than it is height.

import cv2
import numpy as np
from collections import namedtuple
from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure
import imutils
from imutils import perspective


# the named tupled to store the license plate
LicensePlate = namedtuple("LicensePlateRegion", ["success", "plate", "thresh", "candidates"])

#success: a boolean indicating success
#plate: an image of the detected plate
#thresh: the thresholded license plate region, characters on the background
#candidates: a list of character candidates that should be passed on to the ML classifier for final identification

class PlateDetector:
    def __init__(self, image, minPlateW=60, minPlateH=20, numChars=7, minCharW=40):
        # safe to say numChars=7 as the plates in Greece have 3 letters and 4 numbers
        self.image = image
        self.minPlateW = minPlateW
        self.minPlateH = minPlateH
        self.numChars = numChars
        self.minCharW = minCharW
    
    def detect(self):
        # detect license plate regions in the image
        plateRegions = self.detectPlates()
        
        # loop over the license plate regions
        for plateRegion in plateRegions:
            # detect character candidates in the current license plate region
            lp = self.detectCharacterCandidates(plateRegion)
            
            # continue if characters detected
            if lp.success:
                #candidates into characters
                chars = self.scissor(lp)
                
                # yield a tuple of the license plate region and the characters
                yield (plateRegion, chars)

    def detectPlates(self):
    # rectangular and square kernels will be applied to the image, and initialize the list of license plate regions
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))  # almost 3x as wide as its tall
        squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        regions = []
        
        # convert the image to grayscale, and apply the blackhat operation to reveal dark regions such as characters
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
        
        # find the light regions in the image
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKernel)
        # threshold the image to obtain a binary image with an intensity >50 set to 255, otherwise 0
        light = cv2.threshold(light, 50, 255, cv2.THRESH_BINARY)[1]
        
        # Sobel gradient to see vertical changes in gradient such as characters
        # and put image into the range [0, 255]
        gradX = cv2.Sobel(blackhat,
                          ddepth=cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F,
                          dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
                          
        # blur the gradient, apply a closing operation and threshold the image using Otsu's method as Sobel gives noisy images
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                          
        # erosions and dilations on the image
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
                          
        # bitwise 'and' between the 'light' regions of the image to keep only thresholded regions and more erosions and dilations
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)
                          
        # find contours in the thresholded image
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
                          
        # loop over the contours
        for c in cnts:
            # grab the bounding box associated with the contour and compute the area and aspect ratio
            (w, h) = cv2.boundingRect(c)[2:]
            aspectRatio = w / float(h)
                              
            # compute the rotated bounding box of the region
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.cv.BoxPoints(rect)) if imutils.is_cv2() else cv2.boxPoints(rect)
                              
            # make sure the aspect ratio, height, and width of the bounding box fall within reasonable limits
            if (aspectRatio > 3 and aspectRatio < 5) and h > self.minPlateH and w > self.minPlateW:
                # update the list
                regions.append(box)
                              
            # return the list of license plate regions
            return regions

    def detectCharacterCandidates(self, region):
        # Bird's eye view of the plate is needed as skewness may hurt performance
        plate = perspective.four_point_transform(self.image, region)
    
        # extract the Value component from the HSV color space and apply adaptive thresholding to reveal the characters on the plate
        V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]
        T = threshold_local(V, 29, offset=15, method="gaussian")
        thresh = (V > T).astype("uint8") * 255
        thresh = cv2.bitwise_not(thresh)
        
        # resize the plate region to a fixed size
        plate = imutils.resize(plate, width=400)
        thresh = imutils.resize(thresh, width=400)
        
        # apply connected components analysis and mask to store the locations of the character candidates
        labels = measure.label(thresh, neighbors=8, background=0)
        charCandidates = np.zeros(thresh.shape, dtype="uint8")
        
        # loop over the unique components
        for label in np.unique(labels):
            # ignore if it is a background label
            if label == 0:
                continue
            
            # otherwise, construct the label mask to display only connected components for the current label, then find contours in the label mask
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            
            # ensure there is at least one contour in the mask
            if len(cnts) > 0:
                # grab the largest contour which corresponds to the component in the mask, then
                # grab the bounding box for the contour
                c = max(cnts, key=cv2.contourArea)
                (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
                
                # compute the aspect ratio, solidity, and height ratio
                aspectRatio = boxW / float(boxH)
                solidity = cv2.contourArea(c) / float(boxW * boxH)
                heightRatio = boxH / float(plate.shape[0])
                
                # determine if the aspect ratio, solidity, and height of the contour pass tests
                # the parameters are specifically chosen for this dataset, they might change for another dataset
                AspectRatio = aspectRatio < 1.0
                Solidity = solidity > 0.15
                Height = heightRatio > 0.4 and heightRatio < 0.95
                
                # check if the component passes all the tests
                if AspectRatio and Solidity and Height:
                    # compute the convex hull of the contour and draw it on the character
                    # candidates mask
                    hull = cv2.convexHull(c)
                    cv2.drawContours(charCandidates, [hull], -1, 255, -1)
    
        # clear pixels that touch the borders of the character candidates mask and detect contours in the candidates mask
        charCandidates = segmentation.clear_border(charCandidates)
        cnts = cv2.findContours(charCandidates.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
                                
        # if there are more character candidates than the supplied number, prune the candidates
        if len(cnts) > self.numChars:
            (charCandidates, cnts) = self.pruneCandidates(charCandidates, cnts)
                                
        # return the license plate region object containing the license plate, the thresholded
        # license plate, and the character candidates
        return LicensePlate(success=len(cnts) == self.numChars, plate=plate, thresh=thresh,
                                                    candidates=charCandidates)

    def pruneCandidates(self, charCandidates, cnts):
    # initialize the pruned candidates mask and the list of dimensions
        prunedCandidates = np.zeros(charCandidates.shape, dtype="uint8")
        dims = []
    
        # loop over the contours
        for c in cnts:
            # compute the bounding box for the contour and update the list of dimensions
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
            dims.append(boxY + boxH)
        
        # convert the dimensions into a NumPy array and initialize the list of differences and selected contours
        dims = np.array(dims)
        diffs = []
        selected = []
        
        # loop over the dimensions
        for i in range(0, len(dims)):
            # compute the sum of differences between the current dimension and all other dimensions, then update the differences list
            diffs.append(np.absolute(dims - dims[i]).sum())
        
        # find the top number of candidates with the most similar dimensions and loop over the selected contours
        for i in np.argsort(diffs)[:self.numChars]:
            # draw the contour on the pruned candidates mask and add it to the list of selected
            # contours
            cv2.drawContours(prunedCandidates, [cnts[i]], -1, 255, -1)
            selected.append(cnts[i])
        
        # return a tuple of the pruned candidates mask and selected contours
        return (prunedCandidates, selected)
    
    def scissor(self, lp):
        # detect contours in the candidates and initialize the list of bounding boxes and
        # list of extracted characters
        cnts = cv2.findContours(lp.candidates.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        boxes = []
        chars = []
                                
        # loop over the contours
        for c in cnts:
            # compute the bounding box for the contour while maintaining the minimum width
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
            dX = min(self.minCharW, self.minCharW - boxW) // 2
            boxX -= dX
            boxW += (dX * 2)
                                        
            # update the list of bounding boxes
            boxes.append((boxX, boxY, boxX + boxW, boxY + boxH))

        # sort the bounding boxes from left to right
        boxes = sorted(boxes, key=lambda b:b[0])

        # loop over the started bounding boxes
        for (startX, startY, endX, endY) in boxes:
        # extract the ROI from the thresholded license plate and update the characters list
            chars.append(lp.thresh[startY:endY, startX:endX])
        
        # return the list of characters
        return chars

@staticmethod
def preprocessChar(char):
    # find the largest contour in the character, grab its bounding box, and crop it
    cnts = cv2.findContours(char.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) == 0:
        return None
        c = max(cnts, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)
        char = char[y:y + h, x:x + w]

    # return the processed character
    return char


```

The second block of codes below is used to create a descriptor which will be used later in both training and recognizing steps.

```python

# BBPS divides an image into non-overlapping MxN pixel blocks. For each of these blocks, 
# the ratio of foreground (which is thresholded character) pixels to the number of pixels 
# in each block is calculated

import cv2
import numpy as np

class BlockBinaryPixelSum:
    def __init__(self, targetSize=(30, 15), blockSizes=((5, 5),)):
        #the discriminability of the descriptor can be increased by using multiple blockSizes
        self.targetSize = targetSize
        self.blockSizes = blockSizes
    
    def describe(self, image):
        # the image is presumed as a binary image; characters have intensity >0 and the background is 0
        image = cv2.resize(image, (self.targetSize[1], self.targetSize[0]))
        features = []
        
        # loop over the block sizes
        for (blockW, blockH) in self.blockSizes:
            # loop over the image for the current block size
            for y in range(0, image.shape[0], blockH):
                for x in range(0, image.shape[1], blockW):
                    # extract the ROI, count the total number of non-zero pixels normalizing by the total size of the block
                    roi = image[y:y + blockH, x:x + blockW]
                    total = cv2.countNonZero(roi) / float(roi.shape[0] * roi.shape[1])
                    
                    # update the feature vector
                    features.append(total)

        # return the features
        return np.array(features)

```

The third block of codes below is used to train the letter and number classifiers.

```python

from __future__ import print_function
import cv2
import argparse
import pickle
from plate_localization.descriptor import BlockBinaryPixelSum
from sklearn.svm import LinearSVC
from imutils import paths
import imutils

# argument parser
ap = argparse.ArgumentParser()
ap.add_argument"--examples", required=True, help="path to the examples of letters and numbers dataset")
ap.add_argument("--letter-classifier", required=True,
                help="path to the output letter classifier")
ap.add_argument("--number-classifier", required=True,
                help="path to the output number classifier")
args = vars(ap.parse_args())

# initialize characters string
characters  = "abcdefghijklmnopqrstuvwxyz0123456789"

# initialize the data and labels for the letters and numbers
lettersData = []
lettersLabels = []
numbersData = []
numbersLabels = []

# initialize the descriptor
print("describing examples...")
blockSizes = ((5, 5), (5, 10), (10, 5), (10, 10))
# multiple block sizes are used to increase the discriminabilty of the descriptor
desc = BlockBinaryPixelSum(targetSize=(30, 15), blockSizes=blockSizes)

# loop over the example paths
for examplePath in paths.list_images(args["examples"]):
    # load the example image, convert it to grayscale and threshold it
    example = cv2.imread(examplePath)
    example = cv2.cvtColor(example, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(example, 128, 255, cv2.THRESH_BINARY_INV)[1]
    
    # detect contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # sort them from left to right
    cnts = sorted(cnts, key=lambda c:(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[1]))
    
    # loop over the contours
    for (i, c) in enumerate(cnts):
        # grab the bounding box for the contour, extract the ROI, and extract features
        (x, y, w, h) = cv2.boundingRect(c)
        roi = thresh[y:y + h, x:x + w]
        features = desc.describe(roi)
        
        # check if it's a letter
        if i < 26:
            lettersData.append(features)
            lettersLabels.append(characters[i])
        
        # otherwise, it's a number
        else:
            numbersData.append(features)
            numbersLabels.append(characters[i])

# train the letter classifier
print("fitting letter model...")
letterModel = LinearSVC(C=1.0, random_state=61)
letterModel.fit(lettersData, lettersLabels)

# train the digit classifier
print("fitting digit model...")
numberModel = LinearSVC(C=1.0, random_state=61)
numberModel.fit(numbersData, numbersLabels)

# save the letter classifier
print("saving letter model...")
f = open(args["letter_classifier"], "wb")
f.write(pickle.dumps(letterModel))
f.close()

# save the number classifier
print("saving number model...")
f = open(args["number_classifier"], "wb")
f.write(pickle.dumps(numberModel))
f.close()

```

Finally, the fourth and the last block of codes below is used as a main driver script to actually recognize the plate with the help of the first three code blocks above.

```python

from __future__ import print_function
import cv2
import numpy as np
import argparse
import pickle
import imutils
from imutils import paths
from plate_localization.localization import PlateDetector
from plate_localization.descriptor import BlockBinaryPixelSum

# argument parsers
ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True,
                help="path to the images to be classified")
ap.add_argument("--letter-classifier", required=True,
                help="path to the output letter classifier")
ap.add_argument("--number-classifier", required=True,
                help="path to the output number classifier")
args = vars(ap.parse_args())

# load the classifiers
letterModel = pickle.loads(open(args["letter_classifier"], "rb").read())
numberModel = pickle.loads(open(args["number_classifier"], "rb").read())

# initialize the descriptor
blockSizes = ((5, 5), (5, 10), (10, 5), (10, 10))
desc = BlockBinaryPixelSum(targetSize=(30, 15), blockSizes=blockSizes)

# loop over the images
for imagePath in sorted(list(paths.list_images(args["images"]))):
    # load the image
    print(imagePath[imagePath.rfind("/") + 1:])
    image = cv2.imread(imagePath)
    
    # if the width is greater than 600 pixels, resize the image
    if image.shape[1] > 600:
        image = imutils.resize(image, width=600)

    # initialize the plate detector
    pd = PlateDetector(image, numChars=7)
    plates = pd.detect()

    # loop over the detected plates
    for (plateBox, chars) in plates:
    # restructure plateBox
        plateBox = np.array(plateBox).reshape((-1, 1, 2)).astype(np.int32)
    
        # initialize the text containing the recognized characters
        text = ""
        
        # loop over each character
        for (i, char) in enumerate(chars):
            # preprocess the character and describe it
            #char = PlateDetector.preprocessChar(char)
            if char is None:
                continue
            features = desc.describe(char).reshape(1, -1)
            
            # if this is the first 3 characters, use the letter classifier
            if i < 3:
                prediction = letterModel.predict(features)[0]

            # otherwise, use the number classifier
            else:
                prediction = numberModel.predict(features)[0]
            
            # update the text of recognized characters
            text += prediction.upper()
        
        # only draw the characters and bounding box if there are some characters that
        # we can display
        if len(chars) > 0:
            # compute the center of the license plate bounding box
            M = cv2.moments(plateBox)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # draw the license plate region and license plate text on the image
            cv2.drawContours(image, [plateBox], -1, (0, 0, 255), 2)
            cv2.putText(image, text, (cX, cY - 20), cv2.FONT_HERSHEY_DUPLEX, 1.0,
                        (255, 255, 0), 2)

        # display the output image
        cv2.imshow("image", image)
        cv2.waitKey(0)

```

Here is an example of license plate recognized wrongly.

![2019-4-23-License-Plate-Recognition](/images/wrong_plate_recognition.png "2019-4-23-License-Plate-Recognition")
