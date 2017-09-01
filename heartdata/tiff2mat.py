# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 16:26:56 2017

@author: Zlab-6
"""

import scipy.io as sio
from skimage import io
im=io.imread(r'Z:\Lian\DeepLearning\heartdata\2016-6-17\lava\S02-1\SHR_S02-la-4.5-5-5.5-20ms-100%_OD_U-3D_ 4x 0_R02.tiff')
im1=list(im)

a=dict(enumerate(im1,1))
d = {str(k):v for k,v in a.items()}
sio.savemat(r'Z:\Lian\DeepLearning\heartdata\2016-6-17\lava\S02-1\Volumedata.mat', d)


