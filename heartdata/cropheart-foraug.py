# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 03:34:12 2017

@author: qinxi
"""
from PIL import Image
import os
from glob import glob
import numpy as np
import re
import shutil


def get_bound():
    abcd=[]
    folders1=glob('Z:\\Lian\\DeepLearning\\heartdata\\finished\\*\\')  
    for i in folders1:
        folders2=glob(i+'\\*\\')
        for i1 in folders2:
            folders3=glob(i1+'\\*\\')
            for i2 in folders3:
                files=glob(i2+'\\*')
                for i3 in files:
                    print(os.path.basename(i3))
                    i3=i3.replace('\\','/')
                    if not os.path.splitext(i3)[1] in ['.tif','.tiff']:
                        continue
                    else:
                        if 'Labels' in i3:
                            im=Image.open(i3)
                        else:
                            img=Image.open(i3)
                                       
                for j in range(im.n_frames):                                               #search for the first frame has mask
                    im.seek(j)
                    if not im.getbbox():
                        continue
                    else:
                        break 
                        
                a,b,c,d=im.getbbox()
                h=d-b
                for k in range(j,im.n_frames):                                                   #get the largest bound box in the whole file
                    if k%1000 == 0:
                        print(k)
                    im.seek(k)
                    img.seek(k)
                    if not im.getbbox():
                        continue
                    else:       
                        a1,b1,c1,d1=im.getbbox()
                        h1=d1-b1
                        if h1>h:
                            h=h1
#                        if a1<a:
#                            a=a1
#                        if b1<b:
#                            b=b1    
#                        if c1>c:
#                            c=c1
#                        if d1>d:
#                            d=d1
                x=min(a,im.size[0]-c)
                abcd.append([h,x])
    abcd=np.asarray(abcd)
    b=min(abcd[:,1])-10
    d=max(abcd[:,3])+10
    bound=np.asarray([b,d])     
    np.save(r'Z:\Lian\DeepLearning\heartdata\finished\bound',bound)
    return b,d

def crop(b,d):
    folders1=glob('Z:\\Lian\\DeepLearning\\heartdata\\finished\\*\\') 
    for i in folders1:
        folders2=glob(i+'\\*\\')
        for i1 in folders2:
            folders3=glob(i1+'\\*\\')
            for i2 in folders3:
                files=glob(i2+'\\*')
                for i3 in files:
                    print(os.path.basename(i3))
                    i3=i3.replace('\\','/')
                    if not os.path.splitext(i3)[1] in ['.tif','.tiff']:
                        continue
                    else:
                        if 'Labels' in i3:
                            im=Image.open(i3)
                        else:
                            img=Image.open(i3)
            
                name1=os.path.basename(os.path.dirname(i))
                name2=os.path.basename(os.path.dirname(i1))
                name3=os.path.basename(os.path.dirname(i2))
                os.makedirs(r'Z:\Lian\DeepLearning\heartdata\finished\extract\{}_{}_{}'.format(name1,name2,name3))

                for l in range(im.n_frames):
                    im.seek(l)
                    img.seek(l)
                    if not im.getbbox():
                        continue
                    if l%1000 == 0:
                        print(l)
                    box=im.getbbox()
                    cen=(box[1]+box[3])/2

                    im1=im.crop((0, cen-100,im.size[0], cen+100))  
                    img1=img.crop((0, cen-100,im.size[0], cen+100))
                    im1.save(r'Z:\Lian\DeepLearning\heartdata\finished\extract\{}_{}_{}\{}_mask.png'.format(name1,name2,name3,format(l,'05')))
                    img1.save(r'Z:\Lian\DeepLearning\heartdata\finished\extract\{}_{}_{}\{}.png'.format(name1,name2,name3,format(l,'05')))
            

def split(*argv):   
    ii=[]
    for arg in argv:
        x=os.path.basename(os.path.dirname(folders[arg]))
        xx=re.findall('[A-Za-z][A-Za-z]+',x)[0]
        xxx=re.findall('S\d+',x)[0]
        
        for i,j in enumerate(folders):
            if ( xx in j) == True :
                if ( xxx in j) == True:
                    ii.append(i)
    
    folder1=list(set([folders[a] for a in ii]))    
    folder=[folders[f] for f in range(len(folders)) if f not in ii ]
    return sorted(folder1),sorted(folder)
    

def foraug(folder1,folder):  
    for i in folder:   
        print(i)
        files=sorted(glob(i+'*.png'))
        for j in files:
            if os.path.basename(j).replace('.png','').isdigit() == True:
                shutil.copy2(j,'Z:/Lian/DeepLearning/heartdata/for augument/train/ori/'+ re.findall('\w+_S\d+_\d',j)[0] + '_' + os.path.basename(j))
            if os.path.basename(j).replace('.png','').isdigit() == False:
                shutil.copy2(j,'Z:/Lian/DeepLearning/heartdata/for augument/y/mask/'+ re.findall('\w+_S\d+_\d',j)[0] + '_' + os.path.basename(j).replace('_mask',''))
    for z in folder1:
        print(z)
        files=sorted(glob(i+'*.png'))
        for j in files:
            if os.path.basename(j).replace('.png','').isdigit() == True:
                shutil.copy2(j,'Z:/Lian/DeepLearning/heartdata/for augument/val-x/ori/'+ re.findall('\w+_S\d+_\d',j)[0] + '_' + os.path.basename(j))
            if os.path.basename(j).replace('.png','').isdigit() == False:
                shutil.copy2(j,'Z:/Lian/DeepLearning/heartdata/for augument/val-y/mask/'+ re.findall('\w+_S\d+_\d',j)[0] + '_' + os.path.basename(j).replace('_mask',''))

                
                
b,d=get_bound()
crop(b,d)


folders=sorted(glob("Z:/Lian/DeepLearning/heartdata/2016-6-17/extract/*/"))

folder1,folder= split(0,39)
foraug(folder1,folder)
