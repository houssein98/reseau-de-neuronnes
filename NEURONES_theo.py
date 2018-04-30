# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 12:34:03 2018

@author:giraudon@
"""

import numpy as np

#Divers variables

#nombre de layers + input + output:
N_layers=20

#nombre d'input :
N_input=1

#nombre d'output :
N_output=101
def sigmoid(x):
    return np.exp(x)/(1+np.exp(x))
    
#nombre d'échantillons test :
N_test=600
        
#pas dans l'algorithme du gradient :
delta=5*10e-2

#nombre d'itérations dans l'algorithme du gradient :
N_iter=15

#Tableau d'entrée et de sortie : 
INPUT=np.zeros((N_input))
OUTPUT=np.zeros((N_output))

#Tableau des sorties de chaque layer :
Z=[]

#Tableau des y :
Y=[]

#Tableau de test  input/output
T_input=np.zeros((N_test,N_input))
T_output=np.zeros((N_test,N_output))

#Tableau des dérivées de l'erreur par rapport à y :
ERR=[]
    
#Un tableau stockant à la n-ième position
#le nombre de neurones du L-ième layer :
N=np.zeros((N_layers),dtype=int)

N[0]=N_input
N[1]=8
N[2]=10
for k in range(3,N_layers-2):
    N[k]=12
N[N_layers-2]=8
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
            a=np.random.uniform()
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
        
#Remplissage des tableaux de test :       
for k in range(N_test):
    a=np.random.uniform()
    T_input[k][0]=(a-0.5)*2*np.pi
    indice_output=int(100*round(0.5*(1+np.sin((a-0.5)*2*np.pi)),2))
    T_output[k][indice_output]=1

#######################################################

for k in range(N_test):
    #Calcul des sorties de chaque layer :
    for j in range(N[0]):
        Y[0][j]=T_input[k][0]
        Z[0][j]=sigmoid(Y[0][j])
    for L in range(1,N_layers):
        for i in range(N[L]):
            Y[L][i]=sum([W[L-1][j][i]*Z[L-1][j] for j in range(N[L-1])])
            Z[L][i]=sigmoid(Y[L][i])        

    for j in range(N[N_layers-1]):
        ERR[N_layers-1][j]=(Z[N_layers-1][j]-T_output[k][j])*Z[N_layers-1][j]*(1-Z[N_layers-1][j])
    for L in range(1,N_layers-2):        
        L=N_layers-1-L
        for j in range(N[L]):
            ERR[L][j]=Z[L][j]*(1-Z[L][j])*sum([ERR[L+1][k]*W[L][j][k] for k in range(N[L+1])])
            
    for L in range(1,N_layers):
        for k in range(N[L]):
            for j in range(N[L-1]):
                for compteur in range(N_iter):
                    W[L-1][j][k]-=delta*ERR[L][k]*Z[L-1][j]
                    
########################################################
                    
#Test véritable !
for j in range(N[0]):
    Y[0][j]=np.pi/6
    Z[0][j]=np.pi/6
for L in range(1,N_layers):
    for i in range(N[L]):
        Y[L][i]=sum([W[L-1][j][i]*Z[L-1][j] for j in range(N[L-1])])
        Z[L][i]=sigmoid(Y[L][i])  
        
print((Z.index(max(Z))/100.-0.5)*2)

