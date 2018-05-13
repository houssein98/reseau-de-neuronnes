import os
from shutil import copyfile

path="D:/Téléchargements/1600_1609/1600_1609"
paths=[path+'/'+p for p in os.listdir(path)]
files=[]

length=[]
dir_length=[]

#on explore tous les dossiers, et on rajoute les fichiers/leurs paths sans les classifier
while len(paths)!=0:

    p=paths.pop(0)

    if os.path.isdir(p):
        
        #compte la taille des dossiers
        if len(os.listdir(p)) not in length:
            length.append(len(os.listdir(p)))
            dir_length.append([p])
        else:
            dir_length[length.index(len(os.listdir(p)))].append(p) 
            
        paths+=[p+'/'+pp for pp in os.listdir(p)]
        
    if os.path.isfile(p):
        files.append(p)
        
images=[] #nom complet des images(catégories)
dirs=[] #dirs[i] = liste des chemins de images[i]
to_remove=["info.txt","m1.jpeg","m2.jpeg"] #fichiers à enlever

for f in files:
    t=f.split('/')[-1]
    
    if t in to_remove:
        continue
    
    t=t[:6]+t[8:] #on fusionne tous les chromosomes appartenant à la même paire
    
    if t not in images:
        images.append(t)
        dirs.append([f])
    else:
        dirs[images.index(t)].append(f)


dataset_path="D:/Téléchargements/1600_1609/chromosomes"

if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)
    
for i in range(len(images)):
    categ,ext=images[i].split('.')
    categ=dataset_path+'/'+categ
    if not os.path.exists(categ):
        os.mkdir(categ)
    for j in range(len(dirs[i])):
        copyfile(dirs[i][j], categ+'/'+str(j)+'.'+ext)

# 
# ##On divise le dataset 70/30% et on enregistre les images
# 
# train_path="D:/Téléchargements/1600_1609/train"
# validation_path="D:/Téléchargements/1600_1609/validation"
# 
# split=0.7 #on split 70/30%
# c_train=0
# c_valid=0
# 
# if not os.path.exists(train_path):
#     os.mkdir(train_path)
#     
# for i in range(len(images)):
#     categ,ext=images[i].split('.')
#     categ=train_path+'/'+categ
#     if not os.path.exists(categ):
#         os.mkdir(categ)
#     for j in range(round(split*len(dirs[i]))):
#         copyfile(dirs[i][j], categ+'/'+str(j)+'.'+ext)
#         c_train+=1
# 
# if not os.path.exists(validation_path):
#     os.mkdir(validation_path)
#     
# for i in range(len(images)):
#     categ,ext=images[i].split('.')
#     categ=validation_path+'/'+categ
#     if not os.path.exists(categ):
#         os.mkdir(categ)
#     for j in range(round(split*len(dirs[i])),len(dirs[i])):
#         copyfile(dirs[i][j], categ+'/'+str(j)+'.'+ext)
#         c_valid+=1