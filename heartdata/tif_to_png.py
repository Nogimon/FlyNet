# -*- coding: utf-8 -*-
"""
Created on Fri May 19 16:28:15 2017

@author: Zlab
"""

import numpy as np
from skimage import io
im = io.imread('Z:/Lian/DeepLearning/heartdata/S02_01.tif')
for i in range(500):
    io.imsave('Z:/Lian/DeepLearning/heartdata/S02/'+str(i)+'gt.png',np.uint32(im[i]))

im1 = io.imread('Z:/Lian/DeepLearning/heartdata/S02_original_1.tif')
for i in range(500):
    io.imsave('Z:/Lian/DeepLearning/heartdata/S02/'+str(i)+'.png',im1[i])
