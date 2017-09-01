# -*- coding: utf-8 -*-
"""
Created on Fri May 19 14:57:33 2017

@author: L
"""
import numpy as np
'''
from PIL import Image
im = Image.open("Z:/Lian/DeepLearning/heartdata/S02/S02.tiff")
im.show()
import numpy
imarray = numpy.array(im)

imarray.shape(44, 330)
im.size(330, 44)
'''
from skimage import io
im = np.array(io.imread('Z:/Lian/DeepLearning/heartdata/S02/S02labels.tif'))
#a=np.loadtxt('Z:/Lian/DeepLearning/heartdata/1.txt')

im1=im[2045:2548,102:230,:,0]

io.imsave('S02_1.tif',im1)
im1_1=im1/65536
im1_1=im1_1*1.1
im1_1=im1_1.astype(int)
io.imsave('S02_01.tif',im1_1)

im2=np.array(io.imread('Z:/Lian/DeepLearning/heartdata/S02/S02.tiff'))
im2=im2[2045:2548,102:230]

io.imsave('S02_original_1.tif',im2)