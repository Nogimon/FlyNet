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
from flynetfunctions import plotresults, getbox
from parameters import Parameters
from skimage import morphology

#import Image
START = 190
END = 370
#im = io.imread(r'/media/zlab-1/Data/Lian/keras/Purepredict/SHR_S02-la-4.5-5-5.5-20ms-100%_OD_U-3D_ 4x 0_R02.tiff')


#im = io.imread('./Purepredict/SHR_HCM1- 1_OD_U-3D_ 4x 0_R01.tiff')
im = io.imread('./Purepredict/SHR_HCM1+5_OD_U-3D_ 4x 0_R01.tiff')


#SHR_put_AD_125um_m_OD_U-3D_ 4x 0_R01/SHR_put_AD_125um_m_OD_U-3D_ 4x 0_R02.tiff')
#gt = io.imread(r'/media/zlab-1/Data/Lian/keras/Purepredict/SHR_S02-la-4.5-5-5.5-20ms-100%_OD_U-3D_ 4x 0_R02.Labels.tif')
im = np.asarray(im[:,START:END,:])
#loadmodel = '/media/zlab-1/Data/Lian/keras/Purepredict/newweights.h5'
loadmodel = '/media/zlab-1/Data/Lian/keras/nData/weights1.h5'

X_test = []
y_test = []
#for i in range(len(im)):
for i in range(2000):
    X_test.append(cv2.resize(im[i], (128, 128), cv2.INTER_LINEAR))
    #y_test.append(cv2.resize(gt[i], (128, 128), cv2.INTER_LINEAR))
X_test = np.asarray(X_test)
X_test=X_test[...,np.newaxis] 
#y_test = np.asarray(y_test)
#y_test = y_test[...,np.newaxis]    

'''
y_test=[]
for i in folder1:
    print(i)
    file=sorted(glob(i+'*.png'))
    for j in file:
        name=os.path.basename(j)
        img=np.asarray(Image.open(j))
        #img = cv2.cvtColor(cv2.imread(j), cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(img, (128, 128), cv2.INTER_LINEAR)
        #resized = img

        if name.replace('.png','').isdigit() == True:
            X_test.append(resized)
        if name.replace('.png','').isdigit() == False:
            y_test.append(resized)

X_test=np.asarray(X_test)
y_test=np.asarray(y_test)
X_test=X_test[...,np.newaxis]
y_test=y_test[...,np.newaxis]
'''



K.set_image_data_format('channels_last') 


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return ( intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

smooth=1.
'''
inputs = Input((128, 128, 1))
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)


model = Model(inputs=[inputs], outputs=[conv10])

model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

model_checkpoint = ModelCheckpoint('/media/zlab-1/Data/Lian/keras/extract/weights1.h5', monitor='val_loss', save_best_only=True)

earlystop = EarlyStopping(monitor='val_loss', patience=10, mode='auto')

model.fit(train, y, batch_size=32, epochs=150, verbose=1, shuffle=True,
              callbacks=[model_checkpoint, earlystop],validation_data=(X_test, y_test))

'''

X_test = X_test[1100:1500]

model = load_model(loadmodel, custom_objects={'dice_coef_loss':dice_coef,'dice_coef':dice_coef})

a=model.predict(X_test, batch_size=32, verbose=2)




print('Total Number:',len(a))


figuredirectory = '/media/zlab-1/Data/Lian/keras/Purepredict/purepredictresult/'

calculatearea(a, figuredirectory)




def calculatearea(a, directory):
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
        dst = c
        cv2.imwrite(figuredirectory+'{}.jpg'.format(format(i,'05')),dst)
        cv2.imwrite(figuredirectory+'{}_original.jpg'.format(format(i,'05')),X_test[i])
        
        #b1=np.squeeze(b)
        #count1=np.sum(b1)/255
        #count1 = ((250<b1)&(b1<260)).sum()
        #counto.append(count1)

        #vertidiameter, horidiamter = getbox(b1)
        #countvdo.append(vertidiameter)
        #counthdo.append(horidiamter)

        
        c1=np.squeeze(c)

        
        #if name == 'larva':
        c1 = morphology.remove_small_objects(c1, 350)
        

        count3 = ((150<c1)&(c1<260)).sum()
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

    #parameters = Parameters()
    #counto = counto * parameters.yfactor * parameters.xfactor
    countp2 = countp2 * parameters.yfactor * parameters.xfactor
    diametervd = diametervd * parameters.yfactor
    diameterhd = diameterhd * parameters.xfactor

    print(np.average(iou))
    np.save(figuredirectory + 'markresult.npy', countp)
    np.save(figuredirectory +  'iou.npy', iou)


    plt.figure(num = None, figsize = (12, 6), dpi = 200)
    plt.plot(countp2)
    #plt.show()
    plt.savefig('./Purepredict/predictresult.png')
    plt.gcf().clear()

    return(countp2, diametervd, diameterhd)
    
    