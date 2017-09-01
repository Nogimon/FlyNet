# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 11:11:03 2017

@author: qinxi
"""
from PIL import Image
import numpy as np
import os
from glob import glob
import cv2
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from skimage import transform
from prepare_data import prepare_data


train = []
y = []
X_test = []
y_test = []

#Data prepartion Stage and parameter settings
directory = "/media/zlab-1/Data/Lian/keras/nData"
folderstart = 12
folderend = 14
target = 'A'


train, y, X_test, y_test = prepare_data(target, train, y, X_test, y_test)



K.set_image_data_format('channels_last') 


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

smooth=1.

inputs = Input((128,128, 1))
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

model_checkpoint = ModelCheckpoint(directory+'/weights1.h5', monitor='val_loss', save_best_only=True)

earlystop = EarlyStopping(monitor='val_loss', patience=5, mode='auto')

model.fit(train, y, batch_size=32, epochs=150, verbose=1, shuffle=True,
              callbacks=[model_checkpoint, earlystop],validation_data=(X_test, y_test))

a=model.predict(X_test, batch_size=32, verbose=2)

print('Total Number:',len(a))
for i in range(len(a)):
    c=a[i]*255
    c=c.astype('uint8')
    b=y_test[i]*255
    bb=cv2.cvtColor(b,cv2.COLOR_GRAY2RGB)
    cc=cv2.cvtColor(c,cv2.COLOR_GRAY2RGB)
    bbb=np.where(bb>0)
    ccc=np.where(cc>0)
    
    cc[ccc[0:2]]=[255, 0, 0]
    bb[bbb[0:2]]=[0, 0, 255]   
    
    dst = cv2.addWeighted(cc,0.5,bb,0.5,0)    
    cv2.imwrite('/media/zlab-1/Data/Lian/keras/result/{}.jpg'.format(format(i,'05')),dst)
    cv2.imwrite('/media/zlab-1/Data/Lian/keras/result/{}_original.jpg'.format(format(i,'05')),X_test[i])

'''   
cv2.imshow('img',backtorgb)
cv2.waitKey (0)
cv2.imshow('img',b)
cv2.waitKey (0)
cv2.imshow('dst',dst)
cv2.waitKey(0)
'''