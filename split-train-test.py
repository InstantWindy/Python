import glob
import os,shutil
import numpy as np
files ='IMG_4996'
X_list = glob.glob(os.path.join('E:\\Python\\img',files, '*.jpg'))

idx = list(range(len(X_list)))
np.random.shuffle(idx)

dst_path = os.path.join('E:\\Python\\img\\test',files)
if not os.path.exists(dst_path):
    os.makedirs(dst_path)
  

count =0
for i in idx:
    if not os.path.isfile(X_list[i]):
        print ("%s not exist!"%(X_list[i]))
    else:
        filename = X_list[i].split("\\")[-1] #  os.path.basename(X_list[i])
        dstfile = os.path.join(dst_path,filename)
        shutil.move(X_list[i],dstfile)          #移动文件
        print ("move %s -> %s"%( X_list[i],dstfile))
    
    if count == int(len(X_list)*0.2):
        break
    count+=1
        
  
    
