# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 15:16:47 2017

@author: Zlab-6
"""
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image

K.set_image_data_format('channels_last') 
smooth=1.

def iou(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) -intersection + smooth)


def iou_loss(y_true, y_pred):
    return -iou(y_true, y_pred)


def get_model():
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
    return model


data_gen_args = dict(
#    rotation_range=90.,
#    width_shift_range=0.1,
#    height_shift_range=0.2,
#    fill_mode='nearest',
#    cval=0.,
    horizontal_flip=True,
    vertical_flip=True,
    data_format=K.set_image_data_format('channels_last') )
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 1

image_generator = image_datagen.flow_from_directory(
    '/media/zlab-1/Data/Lian/keras/for augument/train',
    class_mode=None,classes=['ori'],
    seed=seed,color_mode='grayscale',
    target_size=(128,128))

mask_generator = mask_datagen.flow_from_directory(
    '/media/zlab-1/Data/Lian/keras/for augument/y',
    class_mode=None,classes=['mask'],
    seed=seed,color_mode='grayscale',
    target_size=(128,128))

from itertools import izip
train_generator = izip(image_generator,mask_generator)

#x=mask_generator.next()
#c=np.uint8(x[15])
#cc=c[...,0]
#Image.fromarray(cc)
#
#y=image_generator.next()
#v=np.uint8(y[31])
#vv=v[...,0]
#Image.fromarray(vv)













    
model=get_model()
model.compile(optimizer=Adam(lr=1e-5), loss=iou_loss, metrics=[iou])

#model = load_model('model-testing-1.h5', custom_objects={'dice_coef_loss': dice_coef_loss,'dice_coef' : dice_coef})
#callbacks = [
#    EarlyStopping(monitor='val_loss', patience=3, verbose=1),
#    ModelCheckpoint('Z:/Lian/DeepLearning/heartdata/model-testing-6.h5', monitor='val_loss', #save_best_only=True),]

histroy=model.fit_generator(
    train_generator,
    steps_per_epoch=30,
    epochs=50,verbose=1,
    #callbacks=callbacks,validation_data=(X_test, y_test)
    )

