---
layout: post
title: Image Classification
---
![2019-4-24-Image-Classification](/images/image_classification_output.png "2019-4-24-Image-Classification")

Keras library has many pre-trained models on the ImageNet dataset of more than 1000 different everyday objects. All we need to do is to download the weights of the models of interest and run the associated code for our dataset. That’s exactly what we do in this post.

We can have different probabilities for different pre-trained networks. For example, both ResNet50 and Xception networks give the “liner” label in the first place among the Top-5 predictions, but with different probabilities of 95% and 74%, respectively. This comparison of different pre-trained models may also give us a baseline of which model is more reliable or suitable for our dataset; but of course, this needs an elaborate thinking.

Just before wrapping this blog post up I want to share an anecdote about this post:

Not only do models learn from us but we also learn from them. At least, I realized that this was true for myself as a non-native speaker of the English language when I searched for an image containing a “ship” on Google to test the classifier. After running my code, I ended up having the output image with the “liner” label which was more appropriate to name the object in the image. Apparently, computers can classify images better than I could possibly do. :)


**Python Code Block:**

```python

import cv2
import numpy as np
import argparse
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception 
from keras.applications import VGG19
from keras.applications import imagenet_utils #to pre-process images and decode outputs
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img


# argument parsers
ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True, help="path to where the image is")
ap.add_argument("--model", type=str, default="resnet50", help="name of pre-trained network")
args = vars(ap.parse_args())

# create a dictionary that connects model names to their classes for Keras
MODELS = {
    "resnet50": ResNet50,
    "vgg19": VGG19,
    "inceptionV3": InceptionV3,
    "xception": Xception,
}

# the input image shape (224x224 pixels) and pre-processing function
inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

# different models use different input sizes such as the InceptionV3 or Xception networks,
# that's why the input shape should be set to (299x299) instead of (224x224)
# and need a different image processing function
if args["model"] in ("inceptionV3", "xception"):
    inputShape = (299, 299)
    preprocess = preprocess_input

# load the network weights from disk --it may take a while to download
#if it's first time; then,they will be cached and next runs will be faster
print("loading {}...".format(args["model"]))
Network = MODELS[args["model"]]
model = Network(weights="imagenet")

# load the input image and preprocess it
print("loading and pre-processing image...")
image = load_img(args["image"], target_size=inputShape)
image = img_to_array(image)

image = np.expand_dims(image, axis=0)
image = preprocess(image)

# classification
print("classifying image with '{}'...".format(args["model"]))
preds = model.predict(image)
P = imagenet_utils.decode_predictions(preds)

# loop over the predictions and print the Top-5 predictions in terminal
for (i, (imagenetID, label, prob)) in enumerate(P[0]):
    print("{}: {:.2f}%".format(label, prob * 100))

# load the image, print the first prediction on the image, and display
orig = cv2.imread(args["image"])
(imagenetID, label, prob) = P[0][0]
cv2.putText(orig, "Label: {}".format(label), (10, 30),
            cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)


```
*The image was taken from: <http://www.seatrade-cruise.com/media/k2/items/cache/3356e0cd9331018e19a8cee016f484a2_XL.jpg>
