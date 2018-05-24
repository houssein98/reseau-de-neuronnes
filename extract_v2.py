from imutils import paths
import random
import cv2
import time
import os

import numpy as np

# convert the labels from str to vectors
#Labelisation des catégories, on crée un vecteur binaire avec un seul 1 dans la catégorie adéquate, et des 0 sinon
def label_(Y):
    categ=[]
    for y in Y:
        if y not in categ:
            categ.append(y)
    categ=sorted(categ)
    Y_categ=[[0 for i in range(len(categ))] for j in range(len(Y))]
    for i in range(len(Y_categ)):
        Y_categ[i][categ.index(Y[i])]=1
    return Y_categ,categ
    
 
args={}
args["dataset"]="1610_1619"

os.chdir("D:/Téléchargements/1610_1619")


imagePaths = sorted(list(paths.list_images(args["dataset"])))

 
# initialize the data and labels
print("[INFO] loading images...")
data1 = []
data2 = []
data3 = []
labels = []
 
# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)
i=0
t=time.time()
to_skip=["m1.jpeg","m2.jpeg"]
# loop over the input images
for imagePath in imagePaths:
    i+=1
    # extract the class label from the image path and update the
    # labels listc
    label = imagePath.split(os.path.sep)[-1]
    if label in to_skip:
        continue
    if label[0:2]=="m2" :
        label="m1"+label[2:]
    label=label[:6]
    if label=="m1_c25":
        label="m1_c24"
    labels.append(label)
    
    
    
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    data1.append(cv2.resize(image, (28, 28)))
    data2.append(cv2.resize(image, (30, 30)))
    data3.append(cv2.resize(image, (40, 40)))



    if i%1000==0:
        print(i,time.time()-t)
        t=time.time()
    


# scale the raw pixel intensities to the range [0, 1]
data1 = np.array(data1, dtype="float") / 255.0
data2 = np.array(data2, dtype="float") / 255.0
data3 = np.array(data3, dtype="float") / 255.0


print('[INFO] Images loaded')

labels,categ = label_(labels)
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
random.seed(42)
random.shuffle(data1)
random.seed(42)
random.shuffle(data2)
random.seed(42)
random.shuffle(data3)
random.seed(42)
random.shuffle(labels)

split=round(len(data1)*0.7)

trainX1=data1[:split]
testX1=data1[split:]

trainX2=data2[:split]
testX2=data2[split:]

trainX3=data3[:split]
testX3=data3[split:]

trainY=labels[:split]
testY=labels[split:]
 

 

print("[INFO] Loading complete")
np.savez_compressed("Dataset10_28x28",trainX1,trainY,testX1,testY)
np.savez_compressed("Dataset10_30x30",trainX2,trainY,testX2,testY)
np.savez_compressed("Dataset10_40x40",trainX3,trainY,testX3,testY)



print('[INFO] Images SAVED')

#alarme quand le code compile
import winsound
duration = 1000 # millisecond
freq = 440  # Hz
winsound.Beep(freq, duration)

print(categ)