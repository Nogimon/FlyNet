# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 16:35:05 2017

@author: L
"""

from PIL import Image
import numpy as np
import os
from glob import glob
import cv2
from keras.models import Model, load_model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from skimage import io
from flynetfunctions import plotresults, findpeaks
from parameters import Parameters
from skimage import morphology

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return ( intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def getbox(image):
    one = np.where(image > 150)
    if (len(one[0]) != 0):
        vertdiameter = np.max(one[0]) - np.min(one[0])
        horidiameter = np.max(one[1]) - np.min(one[1])
    else:
        vertdiameter = 0
        horidiameter = 0
    i = 0
    if (horidiameter > 80):
        plt.imshow(image)
        plt.savefig("./test.png")
        plt.close('all')
    return (vertdiameter, horidiameter)


def calculatearea(a, directory, name, VERTIRANGE, HORIRANGE):
    #only deal with predicted data
    counto=[]
    countp=[]
    countp2=[]
    iou=[]
    countvdo = []
    countvd = []
    counthdo = []
    counthd = []

    for i in range(len(a)):
        c=a[i]*255
        c=c.astype('uint8')
        #b=y_test[i]*255
        #bb=cv2.cvtColor(b,cv2.COLOR_GRAY2RGB)
        cc=cv2.cvtColor(c,cv2.COLOR_GRAY2RGB)
        #bbb=np.where(bb>0)
        ccc=np.where(cc>0)
        
        cc[ccc[0:2]]=[255, 0, 0]
        #bb[bbb[0:2]]=[0, 0, 255]   
        
        
        #dst = cv2.addWeighted(cc,0.5,bb,0.5,0)    
        
        #b1=np.squeeze(b)
        #count1=np.sum(b1)/255
        #count1 = ((250<b1)&(b1<260)).sum()
        #counto.append(count1)

        #vertidiameter, horidiamter = getbox(b1)
        #countvdo.append(vertidiameter)
        #counthdo.append(horidiamter)

        
        c1=np.squeeze(c)

        
        #if name == 'larva':
        #c1 = morphology.remove_small_objects(c1, 100)
        

        count3 = ((200<c1)&(c1<260)).sum()
        #count3 = (c_mor == True).sum()
        countp2.append(count3)

        vertidiameter, horidiamter = getbox(c1)
        countvd.append(vertidiameter)
        counthd.append(horidiamter)
        
        '''
        #simplesum = (b1)/float(255) + (c1)/float(255)
        simplesum = np.divide(b1,255.) + np.divide(c1,255.)
        overlap = ((1.8 < simplesum))
        union = ((0.9 < simplesum))
        iou.append(overlap.sum()/float(union.sum()))

        #print(Image.getbbox(b1))
        '''

        dst = cv2.resize(c1, (HORIRANGE, VERTIRANGE), cv2.INTER_LINEAR)
        img = cv2.resize(X_test[i], (HORIRANGE, VERTIRANGE), cv2.INTER_LINEAR)
        cv2.imwrite(figuredirectory+'/{}.jpg'.format(format(i,'05')),dst)
        cv2.imwrite(figuredirectory+'/{}_original.jpg'.format(format(i,'05')),img)
        

    #counthdo= np.asarray(counthdo)
    counthd = np.asarray(counthd)
    #countvdo = np.asarray(countvdo)
    countvd = np.asarray(countvd)
    #counto = np.asarray(counto)
    countp2 = np.asarray(countp2)

    '''
    #accuracy_vd = 1 - (countvd - countvdo) / (countvdo.astype(float))
    accuracy_vd = (countvd) / (countvdo.astype(float))
    accuracy_hd = (counthd) / (counthdo.astype(float))
    
  
    diametervd = np.vstack((countvdo, countvd))
    diametervd = np.vstack((diametervd, accuracy_vd))
    diameterhd = np.vstack((counthdo, counthd))
    diameterhd = np.vstack((diameterhd, accuracy_hd))
    '''
    diametervd = countvd
    diameterhd = counthd

    #parameters = Parameters()
    #counto = counto * parameters.yfactor * parameters.xfactor
    countp2 = countp2 * parameters.yfactor * parameters.xfactor
    diametervd = diametervd * parameters.yfactor
    diameterhd = diameterhd * parameters.xfactor

    print(np.average(iou))
    #np.save(figuredirectory + 'predict_markresult.npy', countp)
    #np.save(figuredirectory +  'predict_iou.npy', iou)


    plt.figure(num = None, figsize = (15, 6), dpi = 200)
    #plt.plot(diametervd)
    plt.plot(diameterhd)
    plt.ylabel("diameter")
    plt.xlabel("frames")
    #plt.show()

    plt.savefig(figuredirectory + "/" + name + '.png')
    plt.gcf().clear()

    return(countp2, diametervd, diameterhd)

if __name__ == '__main__':

    #set parameter
    VERTIRANGE = 140
    CROPSTART = 210
    CROPEND = CROPSTART + VERTIRANGE
    HORIRANGE = 100
    name = 'S22'
    #directory = "./Purepredict/larva/" + name + ".tiff"
    
    #Diskstation2
    #directory = "/run/user/1000/gvfs/smb-share:server=128.180.65.173,share=data/Lian/flyheart/newdata/processed/SHR_S13_HCM2+_LA_OD_U-3D_ 4x 0_R01.tiff"
    #Diskstation3
    directory = "/run/user/1000/gvfs/smb-share:server=128.180.65.184,share=home/Zlab-NAS3/Kate/262018/Larva/HCM2+/" + name +  "/SHR_" + name + "_HCM2+_LA_OD_U-3D_ 4x 0_R01.tiff"
    START = 000
    END = 4000

    splitname = directory.split("/")
    newfolder = splitname[-1][0:-5]
    print("The file being processed is:" + newfolder)

    #load data

    im = io.imread(directory)


    #SHR_put_AD_125um_m_OD_U-3D_ 4x 0_R01/SHR_put_AD_125um_m_OD_U-3D_ 4x 0_R02.tiff')
    #gt = io.imread(r'/media/zlab-1/Data/Lian/keras/Purepredict/SHR_S02-la-4.5-5-5.5-20ms-100%_OD_U-3D_ 4x 0_R02.Labels.tif')
    im = np.asarray(im[START:END, CROPSTART:CROPEND, (128 - HORIRANGE)/2:(HORIRANGE - 128)/2])
    #loadmodel = '/media/zlab-1/Data/Lian/keras/Purepredict/newweights.h5'
    loadmodeldir = '/media/zlab-1/Data/Lian/keras/Purepredict/weights_new49.h5'

    X_test = []
    y_test = []
    #for i in range(len(im)):
    for i in range(END-START):
        X_test.append(cv2.resize(im[i], (128, 128), cv2.INTER_LINEAR))
        #y_test.append(cv2.resize(gt[i], (128, 128), cv2.INTER_LINEAR))
    X_test = np.asarray(X_test)
    X_test=X_test[...,np.newaxis] 
    


    K.set_image_data_format('channels_last') 

    parameters = Parameters()

    smooth=1.


    #X_test = X_test[START:END]

    model = load_model(loadmodeldir, custom_objects={'dice_coef_loss':dice_coef,'dice_coef':dice_coef})

    a=model.predict(X_test, batch_size=32, verbose=2)




    print('Total Number:',len(a))
    print("The file being processed is:" + newfolder)


    #figuredirectory = '/media/zlab-1/Data/Lian/keras/Purepredict/purepredictresult/'

    
    figuredirectory = directory[0:-len(splitname[-1])] + newfolder
    if(os.path.exists(figuredirectory) == False):
        os.makedirs(figuredirectory)
    print(figuredirectory)

    countp2, diametervd, diameterhd = calculatearea(a, figuredirectory, name, VERTIRANGE, HORIRANGE)



    heartrate, high_ave, high_std, low_ave, low_std = findpeaks(diameterhd, 30)

    print("Heartrate, EDD, ESD is \n", heartrate, high_ave, low_ave)





    
    