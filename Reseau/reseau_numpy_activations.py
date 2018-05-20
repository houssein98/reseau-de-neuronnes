import random
from math import sqrt

import numpy as np

import pickle

def save(network,path):
    f=open(path,'wb')
    pickle.dump(network,f)
    f.close()

def load(path):
    f=open(path, 'rb')
    r=pickle.load(f)
    f.close()
    return r
     
random.seed(123456789)


    
class NeuralNetwork:
    
    def __init__(self,dim_entree,delta,activ="sigmoid"):
        # random.seed(123456789) #controler l'aléat de w
        self.layers=[np.zeros(dim_entree)] #Les neuronnes dans chaque couche
        self.w=[0] #on remplit l'indice 0, car w commence depuis la première couche
        self.Z=[] #Sortie des neuronnes
        if activ=="sigmoid":
            self.activation=[self.sigmoid]
        else:
            self.activation=[activ]
        self.X=[] #vecteur de l'entrée
        self.Y=[] #vecteur de sortie
        self.delta=delta #pas de la descente
        self.E=0
    ####
    #fonctions d'activation
    def sigmoid(self,Y,der=False):
        if der:
            return Y*(1-Y)
        return np.array([1 if y>400 else np.exp(y)/(1+np.exp(y)) for y in Y])
        
    
    def softmax(self,Y,der=False):
        if der:
            kronecker=np.diag(np.ones(len(Y)))
            dZ=Y*kronecker-np.dot(Y.reshape(len(Y),1),Y.reshape(1,len(Y)))
            return dZ
        return np.exp(Y)/np.sum(np.exp(Y))
        
    def relu(self,Y,der=False): # x ; x>=0 | p*x ; x=<0
        Z=np.zeros(Y.shape)
        pente=0.01
        if der:
            Z[Y<=0]=pente
            Z[Y>0]=1
        else:
            Z[Y>0]=Y[Y>0]
            Z[Y<=0]=pente*Y[Y<=0]
        return Z
        
    ####
    
    #On ajoute de taille n au réseau
    def add_layer(self,n,activ="sigmoide"):
        if activ=="sigmoide":
            activ=self.sigmoid
        self.layers.append(np.zeros(n))
        self.activation.append(activ)
        self.w.append(np.random.rand(n,len(self.layers[-2]))) #w_ij random au début
        
    ####

    #On construit le réseau en lui fournissant un seul vecteur en entrée et en sortie
    def build(self,X,Y):
        #On vérifie que les données en entrée sont compatible avec le réseau
        assert len(X)<=len(self.layers[0])
        assert len(Y)==len(self.layers[-1])
        
        self.X=np.array(X)
        self.Y=np.array(Y)
        #On remplit la première couche
        self.layers[0]=self.X[:]
        #On crée la couche des f(y)
        self.Z=self.layers[:]
    ####

    #Evolution d'un pas du réseau
    def evolve(self):
        #On suppose que seul le dernier layer peut-être softmax
        
        for l in range(1,len(self.layers)):
            self.Z[l-1]=self.activation[l-1](self.layers[l-1])#On remplit les sorties de la couche l-1 (Z^l-1)
            self.layers[l]=self.w[l].dot(self.Z[l-1]) #On recalcule les valeurs des neuronnes y^l_j
            
        self.Z[-1]=self.activation[-1](self.layers[-1])#On remplit la dernière sortie   
        # print(self.Z[-1].shape)
        
        #On calcule l'erreur avec le set des résultats
        self.E=0.5*np.linalg.norm(self.Z[-1]-self.Y)**2
       
            
        #Gradient Descent
        dE=[np.zeros(len(self.layers[l])) for l in range(len(self.layers))]
        if self.activation[-1]!=self.softmax:
            dE[-1]=(self.Z[-1]-self.Y)*self.activation[-1](self.Z[-1],True)
        else:
            dZ=self.softmax(self.Z[-1],True)
            dE[-1]=np.transpose(dZ).dot(self.Z[-1]-self.Y)
            
            
        #backpropagation
       
        for l in range(len(self.layers)-2,0,-1):
            dE[l]=np.dot(np.transpose(self.w[l+1]),dE[l+1])*self.activation[l](self.Z[l],True)
            # print(l,self.activation[l](self.Z[l],True).shape,dE[l+1].shape,l)
            
        #adaptation
       # W_kj =  dE_k1 * Z_1j
        # for l in range(1,len(self.layers)):
        #    
        #     for k in range(len(self.w[l])):
        #         
        #         for j in range(len(self.w[l][k])):
        # 
        #             self.w[l][k][j]-=self.delta*dE[l][k]*self.Z[l-1][j]
       
        for l in range(1,len(self.layers)):
            
            
            self.w[l]-=self.delta*np.dot(dE[l].reshape((len(dE[l]),1)),self.Z[l-1].reshape((1,len(self.Z[l-1]))))
           
    #####
    
    #Entrainement avec une liste de vecteurs
    def train(self,X_train,Y_train):
        self.X_train=X_train #liste des vecteurs d'entrée
        self.Y_train=Y_train #liste des vecteurs de sorties
        self.E_train=np.zeros(len(X_train)) #liste des erreurs de chaque vecteur
        
        assert len(X_train)==len(Y_train) #on vérifie que l'entrée est compatible avec la sortie
        
        
        for i in range(len(X_train)):
            #On reconstruit le réseau pour chaque vecteur, et on évolue d'un pas
            self.build(X_train[i],Y_train[i])
            self.evolve()
            self.E_train[i]=self.E
    ####
    
    #Prédiction d'un vecteur d'entrée avec les poids du réseau
    def predict(self,X):
        #On remplit la première couche
        self.layers[0]=np.array(X)
            
        #Calcul de la sortie
        for l in range(1,len(self.layers)):
            self.Z[l-1]=self.activation[l-1](self.layers[l-1])#On remplit les sorties de la couche l-1 (Z^l-1)
            self.layers[l]=self.w[l].dot(self.Z[l-1]) #On recalcule les valeurs des neuronnes y^l_j
            
        self.Z[-1]=self.activation[-1](self.layers[-1])#On remplit la dernière sortie 
        return self.Z[-1]
    ####
    #Prédiction grossière, on ne veut qu'une seule valeur donc on prendre le maximum.
    def predict_gross(self,X):
        Y=self.predict(X)
        Y_gross=np.zeros(len(Y))
        Y_gross[np.argmax(Y)]=1
        return Y_gross
    ####  
    #Evaluation grossière du réseau sur un dataset
    def eval(self,X_eval,Y_eval):
        c=0
        Y_eval=np.array(Y_eval)
        for i in range(len(X_eval)):
            
            Y_predicted=self.predict_gross(X_eval[i]) #On prédit grossièrement le résultat de chaque vecteur

            if (Y_predicted==Y_eval[i]).all():
                c+=1
                
        print("Précision : ",100*c/len(X_eval),"%")
        return c
    ####                


#Traitement de la base 

f=open("D:/9RAYA/2nd_semestre_OUVERTURE/PROJET DE RECHERCHE/letter/letter-recognition.data",'r')

L=f.read().split('\n')[:-1]


for i in range(len(L)):
    L[i]=L[i].split(',')
    
  # On randomize les indices avant de les diviser entrainement/précision
index=[]
for i in range(len(L)):
    n=random.randrange(0,len(L))
    while n not in index:
        index.append(n)
        n=random.randrange(0,len(L))

X = [[float(L[i][j]) for j in range(1,len(L[i]))] for i in index] #Données
Y = [L[i][0] for i in index]  #Catégories (encore string)
        
def label(Y):
    categ=[]
    for y in Y:
        if y not in categ:
            categ.append(y)
    Y_categ=[[0 for i in range(len(categ))] for j in range(len(Y))]
    for i in range(len(Y_categ)):
        Y_categ[i][categ.index(Y[i])]=1
    return Y_categ


Y=label(Y)

#Dimension des données d'entrée et de sortie
input_dim=len(X[0])
output_dim=len(Y[0])

print("Taille du dataset :",len(X))
print("Nombre de données en entrée :",input_dim)
print("Nombre de données en sortie",output_dim)

#On divise 70/30
split=round(len(X)*0.7)
X_train = X[:split//2]
Y_train = Y[:split//2]

X_eval = X[split:]
Y_eval = Y[split:]

#Création du modèle

b=NeuralNetwork(input_dim,2)
# b.add_layer(8)
b.add_layer(output_dim,b.softmax)

#Entrainement et calcul de la précision
i=0
b.train(X_train,Y_train)

max_prec=0
maxi=0

# while max(b.E_train)>0.1:
#     b.train(X_train,Y_train)
#     i+=1
#     if i%100==0:
#         print(max(b.E_train),min(b.E_train))
#         c=b.eval(X_eval,Y_eval)
#         if c>max_prec:
#             max_prec=c
#             maxi=i

while max(b.E_train)>0.3:
    b.train(X_train,Y_train)
    i+=1
    print('Iteration',i," | ", b.E)
    print(max(b.E_train),min(b.E_train))
    c=b.eval(X_eval,Y_eval)
    
    if c>max_prec:
        max_prec=c
        maxi=i
        save(b,'reseau.txt')
        print('Saved : ',i,c)
    if i%10==0:
        
        b.delta/=sqrt(2) #En modifiant le pas, la précision passe de 20 à 54% après ~140 itérations (même seed)
        print(i,b.delta)
    print('_____')
    
            
