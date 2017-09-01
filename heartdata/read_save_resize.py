# -*- coding: utf-8 -*-
"""
Created on Fri May 19 14:57:33 2017

@author: L
"""
import numpy as np
from skimage import io

PATH='Z:/Lian/DeepLearning/heartdata/S05_good condition/'
#frames chosen from 4096
START = 805
END = 1305
NUMPIC = END-START
#cut 128 pixels out
STARTPXL = 137
ENDPXL = 265
#save data number range
SAVEST = 1300

im = np.array(io.imread(PATH+'S05labels.tif'))
#a=np.loadtxt('Z:/Lian/DeepLearning/heartdata/1.txt')

im1=im[START:END,STARTPXL:ENDPXL,:,0]

#io.imsave('S02_1.tif',im1)
im1_1=im1/65536
im1_1=im1_1*1.1
im1_1=im1_1.astype(int)
#io.imsave('S01_label_cut.tif',im1_1)

im2=np.array(io.imread(PATH+'S05.tiff'))
im2=im2[START:END,STARTPXL:ENDPXL]

#io.imsave('S01_cut.tif',im2)

#im1_1 = io.imread('Z:/Lian/DeepLearning/heartdata/S02_01.tif')
for i in range(0,NUMPIC):
    savenumber=i+SAVEST
    io.imsave(PATH+str(savenumber)+'gt.png',np.uint32(im1_1[i]))
    io.imsave(PATH+str(savenumber)+'.png',im2[i])
#im2 = io.imread('Z:/Lian/DeepLearning/heartdata/S02_original_1.tif')


    

from PIL import Image

for i in range(SAVEST,SAVEST+NUMPIC):
    im=Image.open(PATH+str(i)+'.png')
    imr=im.resize((321,321))
    imr.save(PATH+str(i)+'.png')
    
    im=Image.open(PATH+str(i)+'gt.png')
    imr=im.resize((321,321))
    imr.save(PATH+str(i)+'gt.png')
    