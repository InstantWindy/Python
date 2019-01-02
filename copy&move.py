# -*- coding: utf-8 -*-
#!/usr/bin/python
#test_copyfile.py

import os,shutil

def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print "%s not exist!"%(srcfile)
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
        print "move %s -> %s"%( srcfile,dstfile)

def mycopyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print "%s not exist!"%(srcfile)
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件
        print "copy %s -> %s"%( srcfile,dstfile)

srcfile='/Users/xxx/git/project1/test.sh'
dstfile='/Users/xxx/tmp/tmp/1/test.sh'

mymovefile(srcfile,dstfile)



rootsrc='E:\\数据集\\gtCoarse\\gtCoarse\\train_extra'
rootdst='E:\\数据集\\Cityscapes\\labels\\train_extra'

for root,_,files in os.walk(rootsrc):
    for file in files:
        if 'labelIds' in file.split('_')[-1]:
            isExists=os.path.exists(rootdst)
            if not isExists:
                os.makedirs(rootdst) 
            srcfile=os.path.join(root,file)
            dstfile=os.path.join(rootdst,file)
            shutil.copyfile(srcfile,dstfile)      #复制文件
   
for name in os.listdir(rootsrc):
    if 'labelIds' in name.split('_')[-1]:
        isExists=os.path.exists(rootdst)
        if not isExists:
            os.makedirs(rootdst) 
        srcfile=os.path.join(rootsrc,name)
        dstfile=os.path.join(rootdst,name)
        shutil.copyfile(srcfile,dstfile)      #复制文件