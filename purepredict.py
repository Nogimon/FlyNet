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

#import Image
START = 58
END = 230
#im = io.imread(r'/media/zlab-1/Data/Lian/keras/Purepredict/SHR_S02-la-4.5-5-5.5-20ms-100%_OD_U-3D_ 4x 0_R02.tiff')
im = io.imread('./Purepredict/SHR_S01_L2_WT_24B_NF_OD_U-3D_ 4x 0_R02.tiff')
#SHR_put_AD_125um_m_OD_U-3D_ 4x 0_R01/SHR_put_AD_125um_m_OD_U-3D_ 4x 0_R02.tiff')
#gt = io.imread(r'/media/zlab-1/Data/Lian/keras/Purepredict/SHR_S02-la-4.5-5-5.5-20ms-100%_OD_U-3D_ 4x 0_R02.Labels.tif')
im = np.asarray(im[:,START:END,:])
loadmodel = './nData/weights1.h5'

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
model = load_model(loadmodel, custom_objects={'dice_coef_loss':dice_coef,'dice_coef':dice_coef})

a=model.predict(X_test, batch_size=32, verbose=2)




print('Total Number:',len(a))


#For ground truth version

counto=[]
countp=[]
countp2=[]
iou=[]

for i in range(len(a)):
    c=a[i]*255
    c=c.astype('uint8')
    #b=y_test[i]*255
    #bb=cv2.cvtColor(b,cv2.COLOR_GRAY2RGB)
    #cc=cv2.cvtColor(c,cv2.COLOR_GRAY2RGB)
    #bbb=np.where(bb>0)
    #ccc=np.where(cc>0)
    
    #cc[ccc[0:2]]=[255, 0, 0]
    #bb[bbb[0:2]]=[0, 0, 255]   
    
    #dst = cv2.addWeighted(cc,0.5,bb,0.5,0)    
    #cv2.imwrite('Z:/Lian/DeepLearning/keras/result/{}.jpg'.format(format(i,'05')),dst)
    #cv2.imwrite('Z:/Lian/DeepLearning/keras/result/{}_original.jpg'.format(format(i,'05')),X_test[i])
    '''
    b1=np.squeeze(b)
    b1=b1[:,:,2]
    b1=b1/float(65536)
    b1=b1*255
    count1=np.sum(b1)/255
    counto.append(count1)
    '''
    c1=np.squeeze(c)
    count2=np.sum(c1)/255
    countp.append(count2)
    
    count3 = ((250<c1)&(c1<260)).sum()
    countp2.append(count3)
    '''
    simplesum = b1/255 + c1/255
    overlap = ((1.8 < simplesum))
    union = ((0.9 < simplesum))
    iou.append(overlap.sum()/union.sum())
    '''
#plt.plot(counto)
#plt.plot(countp)
plt.plot(countp2)
#plt.show()
plt.savefig('predictresult.png')
plt.gcf().clear()
#plt.plot(iou)
#plt.show()
#plt.savefig('iou.png')

'''

#for no groud truth version
countp2 = []
for i in range(len(a)):
    c=a[i]*255
    c=c.astype('uint8')
    cc=cv2.cvtColor(c,cv2.COLOR_GRAY2RGB)
    ccc=np.where(cc>0)
    cc[ccc[0:2]]=[255, 0, 0]
    cv2.imwrite('/media/zlab-1/Data/Lian/keras/predict_result/{}_predict.jpg'.format(format(i,'05')),cc)

    cv2.imwrite('/media/zlab-1/Data/Lian/keras/predict_result/{}_original.jpg'.format(format(i,'05')),X_test[i])
    
    c1=np.squeeze(c)
    count3 = ((250<c1)&(c1<260)).sum()
    countp2.append(count3)
    

plt.plot(countp2)
plt.savefig('predictionresult.png')
plt.gcf().clear()
'''


    
    