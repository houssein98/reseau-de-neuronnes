# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 12:34:03 2018

@author: giraudon
"""

import numpy as np
import random as rand
#import matplotlib.pyplot as plt

def sigmoid(x):
    return np.exp(x)/(1+np.exp(x))

#Tableau de test input/output
T_input=[]
T_output=[]

#Remplissage des tableaux de test :
iris_dataset = open("/home/giraudon/Documents/PONTS/COURS/NEURONES/iris_dataset.txt","r")
contenu=iris_dataset.read()
k=0
compteur_colonne=0
compteur_ligne=0
while k<len(contenu):
    T_input.append([0,0,0,0])
    T_output.append([0,0,0])    
    while contenu[k]!="I":
        T_input[compteur_ligne][compteur_colonne]=float(contenu[k]+contenu[k+1]+contenu[k+2])
        compteur_colonne+=1
        k+=4
    compteur_colonne=0
    if contenu[k+5]+contenu[k+6]=="se":
        T_output[compteur_ligne][0]=1
        k+=12
    elif contenu[k+5]+contenu[k+6]=="ve":
        T_output[compteur_ligne][1]=1
        k+=16
    elif contenu[k+5]+contenu[k+6]=="vi":
        T_output[compteur_ligne][2]=1
        k+=15
    compteur_ligne+=1

#Divers variables

#nombre de layers + input + output:
N_layers=5

#nombre d'input :
N_input=4

#nombre d'output :
N_output=3

#taille des hidden layers :
size_hidden_layers=7
    
#nombre d'échantillons test :
N_test=len(T_input)
        
#pas dans l'algorithme du gradient :
delta=2

#nombre d'itérations dans l'algorithme du gradient :
N_iter=10

#nombre de fois qu'on entraine le réseau sur le dataset :
N_test_data=15

#Tableau d'entrée et de sortie : 
INPUT=np.zeros((N_input))
OUTPUT=np.zeros((N_output))

#Tableau des sorties de chaque layer :
Z=[]

#Tableau des y :
Y=[]

#Tableau des dérivées de l'erreur par rapport à y :
ERR=[]
    
#Un tableau stockant à la n-ième position
#le nombre de neurones du L-ième layer :
N=np.zeros((N_layers),dtype=int)

N[0]=N_input
for k in range(1,N_layers-1):
    N[k]=size_hidden_layers
N[N_layers-1]=N_output
    

#Le tableau en 3D des poids du réseau :
W=[]

#Remplissage des poids de manière aléatoire :
s=0
a=0
for L in range(N_layers-1):
    W.append([])
    ERR.append([])
    Z.append([])
    Y.append([])
    for j in range(N[L]):
        W[L].append([])
        ERR[L].append(0)
        Z[L].append(0)
        Y[L].append(0)
        for i in range(N[L+1]):
            a=np.random.uniform()-0.5
            W[L][j].append(a)
            s+=a
        #Normalisation
        for i in range(N[L+1]):
            W[L][j][i]/s
        s=0    
        
ERR.append([])        
Z.append([])
Y.append([])
for j in range(N[N_layers-1]):
    ERR[N_layers-1].append(0)
    Z[N_layers-1].append(0)
    Y[N_layers-1].append(0)                    
            
for n in range(N_test_data):            
            
    #######################################################
    
    #On mélange la liste
    T=np.linspace(0,N_test-1,N_test,dtype='int')
    rand.shuffle(T)
    
    for k in T:    
        #Calcul des sorties de chaque layer :
        for j in range(N[0]):
            #Normalisation et centrage (approximatif) en 0
            Y[0][j]=T_input[k][j]/max(T_input[k])-0.4
            Z[0][j]=Y[0][j]
        for L in range(1,N_layers):
            for i in range(N[L]):
                Y[L][i]=sum([W[L-1][j][i]*Z[L-1][j] for j in range(N[L-1])])
                Z[L][i]=sigmoid(Y[L][i])
        #Backpropagation
        for j in range(N[N_layers-1]):
            ERR[N_layers-1][j]=(Z[N_layers-1][j]-T_output[k][j])*Z[N_layers-1][j]*(1-Z[N_layers-1][j])
        for L in reversed(range(1,N_layers-1)):
            for j in range(N[L]):
                ERR[L][j]=Z[L][j]*(1-Z[L][j])*sum([ERR[L+1][i]*W[L][j][i] for i in range(N[L+1])])
        #Algorithme de descente du gradient
        for L in range(1,N_layers):
            for j in range(N[L-1]):
                for i in range(N[L]):
                    for compteur in range(N_iter):
                        W[L-1][j][i]-=delta*ERR[L][i]*Z[L-1][j]
    
    ########################################################
                        

#Test véritable !
Y[0][0]=5.1
Y[0][1]=3.5
Y[0][2]=1.4
Y[0][3]=0.2

for j in range(N[0]):   
    #Normalisation
    Z[0][j]=Y[0][j]/max(Y[0])-0.4
for L in range(1,N_layers):
    for i in range(N[L]):
        Y[L][i]=sum([W[L-1][j][i]*Z[L-1][j] for j in range(N[L-1])])
        Z[L][i]=sigmoid(Y[L][i])  

if Z[N_layers-1].index(max(Z[N_layers-1]))==0:
    print("iris-setosa")
elif Z[N_layers-1].index(max(Z[N_layers-1]))==1:
    print("iris-versicolor")
elif Z[N_layers-1].index(max(Z[N_layers-1]))==2:
    print("iris-virginica")
