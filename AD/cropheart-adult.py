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

print(os.getcwd())
abcd=[]
folders1=glob(os.getcwd()+'/*/')  
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
		x=min(a,im.size[0]-c)
		abcd.append([h,x])
abcd=np.asarray(abcd)
print(max(abcd[:,0]))
