
# import the necessary packages
import time

from imutils import paths

import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os

w=28
h=28

os.chdir("D:/Téléchargements/1600_1609/")
args={}
args["dataset"]="chromosomes"

 
# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []
 
# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)
i=0
t=time.time()
# loop over the input images
for imagePath in imagePaths:
    i+=1
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (w, h))
 
    data.append(image)
    if i%1000==0:
        print(i,time.time()-t)
        t=time.time()
    
    # extract the class label from the image path and update the
    # labels listc
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0

print('[INFO] Images loaded')

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
random.seed(42)
random.shuffle(data)
random.seed(42)
random.shuffle(label)

split=round(len(data)*0.7)

trainX=data[:split]
testX=data[split:]

trainY=labels[:split]
testY=labels[split:]
 
# convert the labels from str to vectors
#Labelisation des catégories, on crée un vecteur binaire avec un seul 1 dans la catégorie adéquate, et des 0 sinon
def label(Y):
    categ=[]
    for y in Y:
        if y not in categ:
            categ.append(y)
    Y_categ=[[0 for i in range(len(categ))] for j in range(len(Y))]
    for i in range(len(Y_categ)):
        Y_categ[i][categ.index(Y[i])]=1
    return Y_categ
 
 
trainY = label(trainY)
testY = label(testY)
print("[INFO] Loading complete")
np.savez_compressed("Dataset2_"+str(w)+"x"+str(h),trainX,trainY,testX,testY)

print('[INFO] Images SAVED')

#alarme quand le code compile
import winsound
duration = 1000 # millisecond
freq = 440  # Hz
winsound.Beep(freq, duration)
