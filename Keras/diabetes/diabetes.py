import random
import numpy as np


from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder


#Traitement de la base 

f=open("D:/9RAYA/2nd_semestre_OUVERTURE/PROJET DE RECHERCHE/diabetes/pima-indians-diabetes.data.csv",'r')

L=f.read().split('\n')


for i in range(len(L)):
    L[i]=L[i].split(',')
    
  # On randomize les indices avant de les diviser entrainement/précision
index=[]
for i in range(len(L)):
    n=random.randrange(0,len(L))
    while n not in index:
        index.append(n)
        n=random.randrange(0,len(L))

X = np.array([[float(L[i][j]) for j in range(0,len(L[i])-1)] for i in index]) #Données
Y = np.array([L[i][len(L[i])-1] for i in index])  #Catégories (encore string)


# On numérise les catégories en entiers 
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# On transforme les entiers en vecteurs binaires ( (1,0,0) si il appartient à la première catégorie par ex )
Y = np_utils.to_categorical(encoded_Y)

#Dimension des données d'entrée et de sortie
input_dim=len(X[0])
output_dim=len(Y[0])

print("Taille du dataset :",len(X))
print("Nombre de données en entrée :",input_dim)
print("Nombre de données en sortie :",output_dim)

#On divise 70/30
split=round(len(X)*0.7)

X_train = X[:split]
Y_train = Y[:split]

X_acc = X[split:]
Y_acc = Y[split:]




# On crée le modèle
model = Sequential()
model.add(Dense(32, input_dim=input_dim, activation='sigmoid')) #Couche d'entrée

model.add(Dense(32, activation='sigmoid')) #Couche cachée

model.add(Dense(output_dim, activation='sigmoid')) #Couche de sortie


# On le compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# On entraine le modèle
print("Début d'entrainement")
model.fit(X_train, Y_train, epochs=200, batch_size=10,  verbose=0)

# On l'évalue
scores = model.evaluate(X_train, Y_train,verbose=0)
print("Précision sur le dataset d'entrainement :",scores[1]*100)

scores = model.evaluate(X_acc, Y_acc,verbose=0)
print("Précision sur le dataset d'évaluation :",scores[1]*100)

scores = model.evaluate(X, Y,verbose=0)
print("Précision sur le dataset entier :",scores[1]*100)


#alarme quand le code compile
import winsound
duration = 1000  # millisecond
freq = 440  # Hz
winsound.Beep(freq, duration)
