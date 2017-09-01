# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 03:34:12 2017

@author: qinxi
"""

#for git
from PIL import Image
import os
from glob import glob
import numpy as np

def get_range():
    abcd=[]
    for ii in ['EP','lava']:
        folders=glob("/media/zlab-1/Data/QX/{}/*/".format(ii))                                           # EP and lava
        for i in folders:
            print(i)
            if not glob(i+'*.tif'):
                continue
            
            label=glob(i+'*.tif')
            file=glob(i+'*.tiff')
        
            im=Image.open(label[0])
            img=Image.open(file[0])
            for j in range(im.n_frames):                                               #search for the first frame has mask
                print(j)
                im.seek(j)
                if not im.getbbox():
                    continue
                else:
                    break 
                
            a,b,c,d=im.getbbox()
            for k in range(j,j+500):                                                   #get the largest bound box in the whole file
                print(k)
                im.seek(k)
                img.seek(k)
                if not im.getbbox():
                    break
                else:       
                    a1,b1,c1,d1=im.getbbox()
                    if a1<a:
                        a=a1
                    if b1<b:
                        b=b1    
                    if c1>c:
                        c=c1
                    if d1>d:
                        d=d1
            x=min(a,im.size[0]-c)
            abcd.append([a,b,c,d,x])
    
    abcd=np.asarray(abcd)
    b=min(abcd[:,1])-10
    d=max(abcd[:,3])+10
    return b,d

def save(b,d):
    for ii in ['EP','lava']:
        folders=glob("/media/zlab-1/Data/QX/{}/*/".format(ii))                                           # EP and lava
        for i in folders:                                                              #crop and save
            print(i)
            if not glob(i+'*.tif'):
                continue
            
            label=glob(i+'*.tif')
            file=glob(i+'*.tiff')
            os.makedirs("/media/zlab-1/Data/QX/extract/{}".format(os.path.basename(file[0])))
        
            im=Image.open(label[0])
            img=Image.open(file[0])
        
            for j in range(im.n_frames):
                im.seek(j)
                if not im.getbbox():
                    continue
                else:
                    break 
        
            for l in range(j,j+500):
                print(l)
                im.seek(l)
                img.seek(l)
                im1=im.crop((0, b,im.size[0], d))  
                img1=img.crop((0, b,im.size[0], d))
                im1.save("/media/zlab-1/Data/QX/extract/{}/{}_mask.png".format(os.path.basename(file[0]),l))
                img1.save("/media/zlab-1/Data/QX/extract/{}/{}.png".format(os.path.basename(file[0]),l))

                
b,d=get_range()
save(b,d)
