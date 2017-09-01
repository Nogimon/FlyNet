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
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.vgg16 import *
from keras.models import *
from keras.regularizers import l2

from utils.get_weights_path import *
from utils.basics import *
from utils.resnet_helpers import *
from utils.BilinearUpSampling import *


from pylab import *
import sys
from keras_contrib.applications import densenet
from keras.layers import *
from keras.engine import Layer
from keras.applications.imagenet_utils import _obtain_input_shape
import tensorflow as tf

from utils.get_weights_path import *
from utils.basics import *
from utils.resnet_helpers import *
from utils.BilinearUpSampling import *


folders=glob("/media/zlab-1/Data/Lian/keras/EP/*/")
train=[]
y=[]

#for larva
'''
folder1=folders[9:11]#+folders[20:21]
folder=folders[0:9]+folders[11:]
'''
#for pupa
folder1=folders[0:2]
folder=folders[2:3]

print('Testing Folders are',folder1,'\n')

for i in folder:
    print(i)
    file=glob(i+'*.png')
    for j in file:
        name=os.path.basename(j)
        img=np.asarray(Image.open(j))
        #img = cv2.cvtColor(cv2.imread(j), cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(img, (128, 128), cv2.INTER_LINEAR)
        #resized = img

        if name.replace('.png','').isdigit() == True:
            train.append(resized)
        if name.replace('.png','').isdigit() == False:
            y.append(resized)

train=np.asarray(train)
y=np.asarray(y)
train=train[...,np.newaxis]
y=y[...,np.newaxis]

#np.save('C:/Users/qinxi/Desktop/python/fruitfly/x.npy',train)
#np.save('C:/Users/qinxi/Desktop/python/fruitfly/y.npy',y)

#from sklearn.model_selection import KFold
#kf = KFold(n_splits=4,random_state=2017)
#kf.get_n_splits(train)
#for train_index, test_index in kf.split(train):
#    print("TRAIN:", train_index, "TEST:", test_index)
#    X_train, X_test =train[train_index], train[test_index]
#    y_train, y_test = y[train_index], y[test_index]
X_test=[]
y_test=[]
for i in folder1:
    print(i)
    file=glob(i+'*.png')
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




K.set_image_data_format('channels_last') 

'''
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return ( intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)
'''
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

weight_decay = 1e-4

smooth=1.


inputs = Input((128, 128, 1))
conv1_1 = Convolution2D(64, (3, 3), activation='relu', border_mode='same', W_regularizer=l2(weight_decay))(inputs)
conv1_2 = Convolution2D(64, (3, 3), activation='relu', border_mode='same', W_regularizer=l2(weight_decay))(conv1_1)
pool1 = MaxPooling2D((2, 2), strides=(2,2))(conv1_2)

conv2_1 = Convolution2D(128, (3, 3), activation='relu', border_mode='same', W_regularizer=l2(weight_decay))(pool1)
conv2_2 = Convolution2D(128, (3, 3), activation='relu', border_mode='same', W_regularizer=l2(weight_decay))(conv2_1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)

conv3_1 = Convolution2D(256, (3, 3), activation='relu', border_mode='same', W_regularizer=l2(weight_decay))(pool2)
conv3_2 = Convolution2D(256, (3, 3), activation='relu', border_mode='same', W_regularizer=l2(weight_decay))(conv3_1)
conv3_3 = Convolution2D(256, (3, 3), activation='relu', border_mode='same', W_regularizer=l2(weight_decay))(conv3_2)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_3)

conv4_1 = Convolution2D(512, (3, 3), activation='relu', border_mode='same', W_regularizer=l2(weight_decay))(pool3)
conv4_2 = Convolution2D(512, (3, 3), activation='relu', border_mode='same', W_regularizer=l2(weight_decay))(conv4_1)
conv4_3 = Convolution2D(512, (3, 3), activation='relu', border_mode='same', W_regularizer=l2(weight_decay))(conv4_2)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4_3)
#need amendment

conv5_1 = Convolution2D(512, (3, 3), activation='relu', border_mode='same', W_regularizer=l2(weight_decay))(pool4)
conv5_2 = Convolution2D(512, (3, 3), activation='relu', border_mode='same', W_regularizer=l2(weight_decay))(conv5_1)
conv5_3 = Convolution2D(512, (3, 3), activation='relu', border_mode='same', W_regularizer=l2(weight_decay))(conv5_2)
#pool5 = MaxPooling2D(pool_size=(2, 2))(conv5_3)

fc1 = Conv2D(4096, 7, 7, activation='relu', W_regularizer=l2(weight_decay))(conv5_3)

drop6 = Dropout(0.5)(fc1)

fc2 = Conv2D(4096, 1, 1, activation='relu', W_regularizer=l2(weight_decay))(drop6)

drop7 = Dropout(0.5)(fc2)

conv8 = Conv2D(21, 1, 1, init='he_normal', activation='linear', border_mode='valid', subsample=(1, 1), W_regularizer=l2(weight_decay))(drop7)

conv9 = BilinearUpSampling2D(size=(32, 32))(conv8)

model = Model(inputs=[inputs], outputs=[conv9])



#model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

model.compile(optimizer=SGD(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

model_checkpoint = ModelCheckpoint('/media/zlab-1/Data/Lian/keras/EP/weights1.h5', monitor='val_loss', save_best_only=True)

earlystop = EarlyStopping(monitor='val_loss', patience=10, mode='auto')

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
