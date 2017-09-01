# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 17:38:26 2017

@author: Zlab-6
"""

from glob import glob
import numpy as np
from PIL import Image
import os
from skimage import io
import matplotlib.pyplot as plt
import cv2

matresult=r"Z:\Lian\DeepLearning\Xialin\EPS01\data\Heartanalysis\Heartmask\Mask_35.pgm"
matresultdir = r'Z:\Lian\DeepLearning\Xialin\EPS01\data\Heartanalysis\Heartmask'

#folder = sorted(glob('Z:\Lian\DeepLearning\Xialin\2016-12-04_SHR_S02-la-4.5-5-5.5-20ms-100%\GPU_processed\SHR_S02-la-4.5-5-5.5-20ms-100%_OD_U-3D_ 4x 0_R02\Heartanalysis\Heartmask\*.pgm'), key = lambda name: int(name[(str.find(name,'Mask_')+5):-4]))
folder = sorted(glob(os.path.join(matresultdir,'*.pgm')), key = lambda name: int(name[(str.find(name,'Mask_')+5):-4]))

im = io.imread(matresult)
result = []
count = []

for i in folder:
    im = io.imread(i)
    #count1 = (im==255).sum()0
    count.append(int((im==255).sum()))
    im2 = (im==255).astype(int)
    result.append(im2)
    
groundtruthdir = r'Z:\Lian\DeepLearning\Xialin\EPS01\SHR_S01_53748-EPupa-4.5-5-5.5Hz-1ms-50%_OD_U-3D_ 4x 0_R01.Labels.tif'
oridir = r'Z:\Lian\DeepLearning\Xialin\EPS01\data\R01.tiff'

groundtruth = io.imread(groundtruthdir)
ori= io.imread(oridir)
groundtruth = groundtruth[:,:,:,2]

iou_dl = np.load('iou.npy')
mark_dl = np.load('markresult.npy')
mark_dl = mark_dl*200/128



iou = []
markcount = []

ENDFRAME = 250
XCENTER = 63
x1 = 57
xs = XCENTER + x1
ZCENTER = 236
z1 = 40
zs = ZCENTER + z1
ROI = 50


for i in range(ENDFRAME+2):
    gt = groundtruth[2*i+1,:,:]
    #gt = cv2.resize(gt, (284, 529), cv2.INTER_LINEAR)
    #gt = gt[ZCENTER-ROI-1:ZCENTER+ROI,XCENTER-ROI-1:XCENTER+ROI]
    
    
    markct = gt.sum()/65535
    markcount.append(markct)
    
    gt = (gt==65535).astype(int)
    gt = gt[zs-2*ROI-1:zs,xs-2*ROI-1:xs]
    
    mim = result[i]
    
    sumimg = gt + mim
#    a=Image.fromarray(255*np.uint8(mim))
#    b=Image.fromarray(255*np.uint8(gt))
#    c=Image.fromarray(255*np.uint8(sumimg))
#    a.save(r'Z:\Lian\DeepLearning\Xialin\test\{}_mim.jpg'.format(i))
#    b.save(r'Z:\Lian\DeepLearning\Xialin\test\{}_gt.jpg'.format(i))
#    c.save(r'Z:\Lian\DeepLearning\Xialin\test\{}_sumimg.jpg'.format(i))
    
    inter = (sumimg>1.8).sum()
    union = (sumimg>0.9).sum()
    iou.append(inter/union)
    
    
    mark_dl[i] = mark_dl[2*i+1]
    iou_dl[i] = iou_dl[2*i+1]
    
plt.plot(count[0:ENDFRAME])
plt.plot(markcount[0:ENDFRAME])
plt.plot(mark_dl[0:ENDFRAME])
#plt.show()
plt.savefig('markresult_matlab.png')
plt.gcf().clear()

plt.plot(iou[0:ENDFRAME])
plt.plot(iou_dl[0:ENDFRAME])
#plt.show()
plt.savefig('iou.png')

iou_average = np.average(iou[:ENDFRAME])
print('IOU is:',iou_average)


