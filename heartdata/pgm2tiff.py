# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 15:57:31 2017

@author: Zlab-6
"""


from glob import glob
import numpy as np
from tifffile import imsave
from PIL import Image
import os

inputdir=r"Z:\Data\Drosophila\HFD_CLOCK experiments\Round3\CLOCK_RNAi_HFD_L3"
outputdir=r"Z:\Lian\DeepLearning\heartdata\2014-10-1"

folder=glob(inputdir +"\\*\\")
for j in folder:
    files=glob(glob(j+'Analyzed\\Mmode\\*\\')[0]+'*.pgm')
    images=[]
    print(j)
    for index,i in enumerate(files):
        a=np.asarray(Image.open(i))
        images.append(a)
        if index%100 == 0:
            print(index)
        
    image = np.asarray(images)
    imsave(outputdir+"\\{}.tif".format(os.path.basename(os.path.dirname(files[0]))), image)

    
    
    
    
    
    
    
    
    
    
#files=glob("Z:\\Lian\\DeepLearning\\heartdata\\2014-9-12\\HFD_L3\\2014-09-12_SHR_S01_HFD_WT_24B_L3\\Analyzed\\Mmode\\SHR_S01_HFD_WT_24B_L3_OD_U-3D_ 4x 0_R01\\*.pgm")
#images=[]
#for index,i in enumerate(files):
#    a=np.asarray(Image.open(i))
#    images.append(a)
#    print(index)
#    
#image = np.asarray(images)
#imsave('Z:\\Lian\\DeepLearning\\heartdata\\2014-9-12\\HFD_L3\\2014-09-12_SHR_S01_HFD_WT_24B_L3\\2014-09-12_SHR_S01_HFD_WT_24B_L3.tif', image)
#
