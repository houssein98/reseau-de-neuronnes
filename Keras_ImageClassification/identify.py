# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os
from imutils import paths



os.chdir("D:/Téléchargements/1600_1609/")

args={}
args["patient"]="1600_1609/1601-458/1/1.kar"
args["image"]="0.jpeg"
args["model"]="chromosomes.model"

categ=os.listdir("chromosomes")

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])

def indice_max(L):
    return np.where(L==np.max(L))[0][0]

def predict(image_path):
    # load the image
    image = cv2.imread(image_path)
    orig = image.copy()
    # pre-process the image for classification
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # classify the input image
    chromo = model.predict(image)[0]
    # # build the label
    label = categ[indice_max(chromo)]
    proba = np.max(chromo)
    label = "{}: {:.2f}%".format(label, proba * 100)
    return label


# 
# label = predict(args["image"])
#  
# # # draw the label on the image
# output = imutils.resize(orig, width=200)
# cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
#     0.7, (0, 255, 0), 2)
#  
# # show the output image
# cv2.imshow("Output", output)
# cv2.waitKey(0)



imagePaths = sorted(list(paths.list_images(args["patient"])))
labels=[]
original=[]
for image in imagePaths:
    labels.append(predict(image))
    original.append(image.split(os.path.sep)[-1])