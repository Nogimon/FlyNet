# -*- coding: utf-8 -*-
"""
Created on Fri May 19 14:57:33 2017

@author: L
"""
import numpy as np
from skimage import io

#frames chosen from 4096
START = 0
END = 300
NUMPIC = END-START
#cut 128 pixels out
STARTPXL = 100
ENDPXL = 228
#save data number range
SAVEST = 1000

im = np.array(io.imread('//128.180.65.173/data/Lian/DeepLearning/heartdata/S04/S04labels.tif'))
#a=np.loadtxt('Z:/Lian/DeepLearning/heartdata/1.txt')

im1=im[START:END,STARTPXL:ENDPXL,:,0]

#io.imsave('S02_1.tif',im1)
im1_1=im1/65536
im1_1=im1_1*1.1
im1_1=im1_1.astype(int)
#io.imsave('S01_label_cut.tif',im1_1)

im2=np.array(io.imread('//diskstation2/data/Lian/DeepLearning/heartdata/S04/S04.tiff'))
im2=im2[START:END,STARTPXL:ENDPXL]

#io.imsave('S01_cut.tif',im2)

#im1_1 = io.imread('Z:/Lian/DeepLearning/heartdata/S02_01.tif')
for i in range(0,NUMPIC):
    savenumber=i+SAVEST
    io.imsave('//diskstation2/data/Lian/DeepLearning/heartdata/S04/'+str(savenumber)+'gt.png',np.uint32(im1_1[i]))
    io.imsave('//diskstation2/data/Lian/DeepLearning/heartdata/S04/'+str(savenumber)+'.png',im2[i])
#im2 = io.imread('Z:/Lian/DeepLearning/heartdata/S02_original_1.tif')


    

from PIL import Image

for i in range(SAVEST,SAVEST+NUMPIC):
    im=Image.open('//diskstation2/data/Lian/DeepLearning/heartdata/S04/'+str(i)+'.png')
    imr=im.resize((321,321))
    imr.save('//diskstation2/data/Lian/DeepLearning/heartdata/S04/'+str(i)+'.png')
    
    im=Image.open('//diskstation2/data/Lian/DeepLearning/heartdata/S04/'+str(i)+'gt.png')
    imr=im.resize((321,321))
    imr.save('//diskstation2/data/Lian/DeepLearning/heartdata/S04/'+str(i)+'gt.png')
    