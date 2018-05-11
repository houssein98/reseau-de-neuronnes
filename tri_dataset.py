import os
from shutil import copyfile

path="D:/Téléchargements/1600_1609/1600_1609"
paths=[path+'/'+p for p in os.listdir(path)]
files=[]

length=[]
dir_length=[]
c_images=0
c_a=0
while len(paths)!=0:

    p=paths.pop(0)

    if os.path.isdir(p):
        if len(os.listdir(p)) not in length:
            length.append(len(os.listdir(p)))
            dir_length.append([p])
        else:
            dir_length[length.index(len(os.listdir(p)))].append(p)
            
        paths+=[p+'/'+pp for pp in os.listdir(p)]
        
    if os.path.isfile(p):
        files.append(p)
        
images=[]
dirs=[]
for f in files:
    t=f.split('/')[-1]
    if t not in images:
        images.append(t)
        dirs.append([f])
    else:
        dirs[images.index(t)].append(f)

# dataset_path="D:/Téléchargements/1600_1609/organized_dataset"
# 
# if not os.path.exists(dataset_path):
#     os.mkdir(dataset_path)
#     
# for i in range(len(images)):
#     categ,ext=images[i].split('.')
#     categ=dataset_path+'/'+categ
#     if not os.path.exists(categ):
#         os.mkdir(categ)
#     for j in range(len(dirs[i])):
#         copyfile(dirs[i][j], categ+'/'+str(j)+'.'+ext)
