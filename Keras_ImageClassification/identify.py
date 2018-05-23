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

width=28
height=28

args={}
args["patient"]="10.kar"
args["image"]="0.jpeg"
args["model"]="chromosomes"+str(w)+"x"+str(h)+".model"

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
    image = cv2.resize(image, (width, height))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # classify the input image
    chromo = model.predict(image)[0]
    # # build the label
    predicted = categ[indice_max(chromo)]
    proba = np.max(chromo)
    label = "{}: {:.2f}%".format(predicted, proba * 100)
    return predicted,chromo,label

def label_(labels):
    # On numérise les catégories en entiers 
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded_labels = encoder.transform(labels)
    # On transforme les entiers en vecteurs binaires ( (1,0,0,...) si il appartient à la première catégorie par ex )
    labels = to_categorical(encoded_labels)
    return labels
    
    
def evaluate_dataset(category_path):
 
    # grab the image paths 
    imagePaths = sorted(list(paths.list_images(category_path)))
    data=[]
    labels=[]
    # loop over the input images
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (width, height))
        image = img_to_array(image)
        data.append(image)
        
        # extract the class label from the image path and update the
        # labels listc
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
        
    
    # scale the raw pixel intensities to the range [0, 1]
    X = np.array(data, dtype="float") / 255.0
    Y = label_(labels)
    print(X.shape)
    print(Y.shape)
    return model.evaluate(X,Y,verbose=0)
    
    

imagePaths = sorted(list(paths.list_images(args["patient"])))
to_remove=["m1.jpeg","m2.jpeg"]
predicted=[]
originals=[]
for image in imagePaths:
    file=image.split(os.path.sep)[-1]
    
    if file in to_remove:
        continue
        
    name=file[:file.find("c")+3]
    originals.append(name)
    
    predicted.append(predict(image)[0])

    
print("Précision : ",np.count_nonzero(np.array(predicted)==np.array(originals))/len(predicted)*100)


