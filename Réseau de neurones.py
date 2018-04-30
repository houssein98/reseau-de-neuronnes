import numpy as np
import numpy.linalg as lg
import random as rd

##ATTENTION AUX PROBLEMES PAR RAPPORT AU NOMBRE DE COUCHES (la numérotation)


delta = 0.5

#Création du réseau de neurones
#On considère ici un réseau avec autant de neurones par couche sur toutes les couches
def f(x):
    return (np.exp(x)/(1+np.exp(x)))

def cree_reseau(q, n):  #Crée un réseau de neurones, q = nombre de couches du réseau de neurones (on veut aller de 0 à q), n = nombre de neurones par couche
    M=np.zeros((q+1,n,n))
    for i in range(q+1):
        for j in range(n):
            for k in range(n):
                M[i, j, k] = rd.random()
    return M


def passe_couche(N, Y):     #Effectue le passage du signal Y (en entrée) par une certaine couche du réseau, N = matrice des poids des synapses de cette couche
    n = N.shape[0]
    Z=np.zeros(n)
    
    for i in range(n):
        for j in range(n):
            Z[i] += N[j, i] * Y[j]
        Z[i] = f(Z[i])
        
    return Z


def evolue_reseau(M, Y0, T):    #On fait passer le signal Y0 dans le réseau M, et on met à jour ce réseau. On s'attend à avoir T en sortie (sert à calculer l'erreur)
    q = M.shape[0] -1
    n = M.shape[1]
    
    #On fait passer le signal dans le réseau
    Ztot = np.zeros((q+1,n))
    Ztot[0,:] = Y0
    Z=Y0
    for k in range(1, q+1):
        Z = passe_couche(M[k], Z)
        Ztot[k, :] = Z
        
    #Calcul de l'erreur entre la sortie et ce qu'on attend
    erreur = 0.5 * np.dot(np.transpose(T-Z), T-Z)
    dE=np.zeros((q+1,n))
    
    #Initialisation de dE
    for j in range(n):
        dE[q,j] = Z[j] * (1-Z[j]) * (Z[j] - T[j])
    
    #Rétropropagation
    for L in range(q-1,0,-1):
        for j in range(n):
            somme_k = 0
            for k in range (n):
                somme_k += dE[L+1, k] * M[L+1, k, j]        ##################################    EST-CE QUE C'EST w_kj OU w_jk ???
            dE[L,j] = Ztot[L,j] * (1- Ztot[L,j]) * somme_k
    
    #Adaptation
    for L in range(1,q+1):
        for k in range(n):
            for j in range(n):
                M[L, k, j] -= delta * dE[L, j] * Ztot[L-1, j]
    
    return M, Z, erreur

########---------------------------------------------------------------------------------------------------------------------------------------#############
### Zone de tests ###

### Test 1
q = 5
n = 10
Y0= [rd.random() for i in range(n)]
T = [rd.random() for i in range(n)]
M = cree_reseau(q, n)

erreur = 100
ind = 0
Z=Y0

while (erreur>0.001):
    ind+=1
    M, Z, erreur = evolue_reseau(M, Y0, T)

print('Nombre d\'itérations effectuées :')
print(ind)
print('Erreur entre l\'entrée et l\'attente :')
print(erreur)
print('Sortie finale :')
print(Z)
print('Vecteur erreur :')
print(abs(T-Z))


### Test 2
q = 2
n = 2
t = 2500    #Nombre de données

Ytest=[]    #Création d'une base de données tagguée
Ttest=[]    #Création des attentes

M=cree_reseau(q, n)     #Création du réseau

for i in range (t) :
    Y0= [10*rd.random() for i in range(n)]
    T = [10*rd.random() for i in range(n)]
    Ytest.append(Y0)
    Ttest.append(T)

err_moy1=0
for k in range(t):      #Apprentissage
    M, Z, erreur = evolue_reseau(M, Ytest[k], Ttest[k])
    print(erreur)
    err_moy1+=erreur

err_moy2 = 0
for k in range(t):      #Essai du réseau
    M, Z, erreur = evolue_reseau(M, Ytest[k], Ttest[k])
    print(erreur)
    err_moy2+=erreur

print('Erreurs moyennes')
print(err_moy1/t)
print(err_moy2/t)

