## adapté de : https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/

# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
 
class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)
 
        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            
        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding="same",
            input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
 
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
 
        # return the constructed network architecture
        return model
        
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

#images dimensions
w=30
h=30

os.chdir("D:/Téléchargements/1600_1609/")
args={}
args["dataset"]="chromosomes"
args["model"]="chromosomes"+str(w)+"x"+str(h)+".model"
args["loss"]="Erreur"+str(w)+"x"+str(h)
args["acc"]="Precision"+str(w)+"x"+str(h)



# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 100
INIT_LR = 1e-3
BS = 32

with np.load("Dataset2_"+str(w)+"x"+str(h)+".npz") as data:
    trainX=data['arr_0']
    trainY=data['arr_1']
    testX=data['arr_2']
    testY=data['arr_3']


                                
# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

                                
# initialize the model
print("[INFO] compiling model...")
model = LeNet.build(width=w, height=h, depth=3, classes=len(trainY[0]))

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])
    
 
# train the network
print("[INFO] training network...")

H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS, verbose=2)

# H = model.fit(trainX,trainY,validation_data=(testX,testY),batch_size=BS,epochs=EPOCHS,verbose=2)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["acc"], label="Entrainement")
plt.plot(np.arange(0, N), H.history["val_acc"], label="Validation")
plt.title("Evolution de la précision en fonction des époques")
plt.xlabel("Epoque #")
plt.ylabel("Précision")
plt.legend(loc="lower right")
plt.savefig(args["acc"])
plt.clf()

plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="Entrainement")
plt.plot(np.arange(0, N), H.history["val_loss"], label="Validation")
plt.title("Evolution de l'entropie croisée en fonction des époques")
plt.xlabel("Epoque #")
plt.ylabel("Erreur")
plt.legend(loc="lower left")
plt.savefig(args["loss"])