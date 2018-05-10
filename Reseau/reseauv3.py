import random
from math import exp

class NeuralNetwork:
    
    def __init__(self,dim_entree,delta):
        # random.seed(6969) #controler l'aléat de w
        self.layers=[[0 for i in range(dim_entree)]] #Les neuronnes dans chaque couche
        self.w=[0] #on remplit l'indice 0, car w commence depuis la première couche
        self.Z=[] #Sortie des neuronnes
        
        self.X=[] #vecteur de l'entrée
        self.Y=[] #vecteur de sortie
        self.delta=delta #pas de la descente
        self.E=0
    ####
    #fonctions d'activation
    def f(self,x):
        return exp(x)/(1exp(x))

    def df(self,x):
        return f(x)*(1-f(x))
    ####
    
    #On ajoute de taille n au réseau
    def add_layer(self,n):
        self.layers.append([0 for i in range(n)])
        
        self.w.append([[random.random() for j in range(len(self.layers[-2]))]for i in range(n)]) #w_ij random au début
        
    ####

    #On construit le réseau en lui fournissant un seul vecteur en entrée et en sortie
    def build(self,X,Y):
        #On vérifie que les données en entrée sont compatible avec le réseau
        assert len(X)<=len(self.layers[0])
        assert len(Y)==len(self.layers[-1])
        
        self.X=X
        self.Y=Y
        #On remplit la première couche
        for i in range(len(X)):
            self.layers[0][i]=X[i]
        #On crée la couche des f(y)
        self.Z=self.layers[:]
    ####

    #Evolution d'un pas du réseau
    def evolve(self):
        for l in range(1,len(self.layers)):
            self.Z[l-1]=[self.f(y) for y in self.layers[l-1]] #On remplit les sorties de la couche l-1 (Z^l-1)
            for j in range(len(self.layers[l])):
                self.layers[l][j]=0
                for k in range(len(self.Z[l-1])):
                    self.layers[l][j]+=self.w[l][j][k]*self.Z[l-1][k] #On recalcule les valeurs des neuronnes y^l_j
                    
        self.Z[-1]=[self.f(y) for y in self.layers[-1]] #On remplit la dernière sortie 
        
        #On calcule l'erreur avec le set des résultats
        self.E=0
        for i in range(len(self.Z[-1])):
            self.E+=0.5*(self.Z[-1][i]-self.Y[i])**2
            
        #Gradient Descent
        
        dE=[[0 for j in range(len(self.layers[i]))] for i in range(len(self.layers))]
        
        for j in range(len(self.layers[-1])):
            dE[-1][j]=(self.Z[-1][j]-self.Y[j])*self.Z[-1][j]*(1-self.Z[-1][j])
            
        #backpropagation
        for l in range(len(self.layers)-2,0,-1):
            for j in range(len(self.layers[l])):
                for k in range(len(self.layers[l+1])):
                    dE[l][j]+=dE[l+1][k]*self.w[l+1][k][j]
                dE[l][j]*=self.Z[l][j]*(1-self.Z[l][j])
                
            
                
        #adaptation
        
        for l in range(1,len(self.layers)):
           
            for k in range(len(self.w[l])):
                
                for j in range(len(self.w[l][k])):
        
                    self.w[l][k][j]-=self.delta*dE[l][k]*self.Z[l-1][j]
    #####
    
    #Entrainement avec une liste de vecteurs
    def train(self,X_train,Y_train):
        self.X_train=X_train #liste des vecteurs d'entrée
        self.Y_train=Y_train #liste des vecteurs de sorties
        self.E_train=[0 for i in range(len(X_train))] #liste des erreurs de chaque vecteur
        
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
        for i in range(len(X)):
            self.layers[0][i]=X[i]
            
        #Calcul de la sortie
        for l in range(1,len(self.layers)):
            self.Z[l-1]=[self.f(y) for y in self.layers[l-1]]  #on calcule les images de chaque neuronne
            for j in range(len(self.layers[l])):
                self.layers[l][j]=0
                for k in range(len(self.Z[l-1])):
                    self.layers[l][j]+=self.w[l][j][k]*self.Z[l-1][k] #on recalcule les neuronnes avec les images actualisées
                    
        self.Z[-1]=[self.f(y) for y in self.layers[-1]] #calcul des images de la dernière couche
        return self.Z[-1]
    ####
    #Prédiction grossière, on ne veut qu'une seule valeur donc on prendre le maximum.
    def predict_gross(self,X):
        Y=self.predict(X)
        Y_gross=[0 for i in range(len(Y))]
        Y_gross[Y.index(max(Y))]=1
        return Y_gross
    ####  
    #Evaluation grossière du réseau sur un dataset
    def eval(self,X_eval,Y_eval):
        c=0
        for i in range(len(X_eval)):
            
            Y_predicted=self.predict_gross(X_eval[i]) #On prédit grossièrement le résultat de chaque vecteur
            
            if Y_predicted==Y_eval[i]:
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
X_train = X[:split]
Y_train = Y[:split]

X_eval = X[split:]
Y_eval = Y[split:]

#Création du modèle

b=NeuralNetwork(input_dim,3)
b.add_layer(16)
b.add_layer(output_dim)

#Entrainement et calcul de la précision
i=0
b.train(X_train,Y_train)

max_prec=0
maxi=0

while max(b.E_train)>0.3:
    b.train(X_train,Y_train)
    i+=1
    print(max(b.E_train),min(b.E_train))
    c=b.eval(X_eval,Y_eval)
    if c>max_prec:
        max_prec=c
        maxi=i
    if i%10==0:
        b.delta/=2
            