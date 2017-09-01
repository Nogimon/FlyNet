# -*- coding: utf-8 -*-
"""
Created on Fri May 19 16:28:15 2017

@author: Zlab
"""

import numpy as np
from skimage import io
from PIL import Image

for i in range(500):
    im=Image.open('Z:/Lian/DeepLearning/heartdata/S02/'+str(i)+'.png')
    imr=im.resize((321,321))
    imr.save('Z:/Lian/DeepLearning/heartdata/nS02/'+str(i)+'.png')
    
    im=Image.open('Z:/Lian/DeepLearning/heartdata/S02/'+str(i)+'gt.png')
    imr=im.resize((321,321))
    imr.save('Z:/Lian/DeepLearning/heartdata/nS02/'+str(i)+'gt.png')
    