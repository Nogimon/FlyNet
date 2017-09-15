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
from keras.models import Model, load_model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from flynetfunctions import plotresults, getbox



train=[]
y=[]

target = 'A'
   
if target=='AD':
    directory = "/media/zlab-1/Data/Lian/keras/AD"
    folders=sorted(glob(directory+"/*/"))
    start1 = 19
    end1 = 20
    folder1 = folders[start1:end1]
    folder=folders[0:start1]+folders[end1:]

else:
    directory = "/media/zlab-1/Data/Lian/keras/nData"
    directory2 = "./AD"
    start1 = 18
    end1 = 19
    start2 = 53
    end2 = 54
    start3 = -11
    end3 = -10
    folders=sorted(glob(directory+"/*/"))+sorted(glob(directory2+"/*/"))
    folder1=folders[start1:end1]+folders[start2:end2] + folders[start3:end3]
    folder=folders[0:start1]+folders[end1:start2]+folders[end2:start3] + folders[end3:]


X_test=[]
y_test=[]
testcount=[]

for i in folder1:
    print(i)
    file=sorted(glob(i+'*.png'))
    print("number of files is ", len(file) / 2)
    testcount.append(len(file) / 2)

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

print(testcount)


K.set_image_data_format('channels_last') 


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return ( intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

smooth=1.

#model = load_model('/media/zlab-1/Data/Lian/keras/EP/weights1.h5', custom_objects={'dice_coef_loss':dice_coef,'dice_coef':dice_coef})
model = load_model(directory+'/weights1.h5', custom_objects={'dice_coef_loss':dice_coef,'dice_coef':dice_coef})

a=model.predict(X_test, batch_size=32, verbose=2)




print('Total Number:',len(a))
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
    
    b1=np.squeeze(b)
    #count1=np.sum(b1)/255
    count1 = ((250<b1)&(b1<260)).sum()
    counto.append(count1)

    vertidiameter, horidiamter = getbox(b1)
    countvdo.append(vertidiameter)
    counthdo.append(horidiamter)

    
    c1=np.squeeze(c)

    #count2=(np.sum(c1))/float(255)
    #countp.append(count2)

    count3 = ((150<c1)&(c1<260)).sum()
    countp2.append(count3)

    vertidiameter, horidiamter = getbox(c1)
    countvd.append(vertidiameter)
    counthd.append(horidiamter)
    
    #simplesum = (b1)/float(255) + (c1)/float(255)
    simplesum = np.divide(b1,255.) + np.divide(c1,255.)
    overlap = ((1.8 < simplesum))
    union = ((0.9 < simplesum))
    iou.append(overlap.sum()/float(union.sum()))

    #print(Image.getbbox(b1))


counthdo= np.asarray(counthdo)
counthd = np.asarray(counthd)
countvdo = np.asarray(countvdo)
countvd = np.asarray(countvd)

#accuracy_vd = 1 - (countvd - countvdo) / (countvdo.astype(float))
accuracy_vd = (countvd) / (countvdo.astype(float))
accuracy_hd = (counthd) / (counthdo.astype(float))

print(counthdo.shape, counthd.shape, accuracy_hd.shape)


diametervd = np.vstack((countvdo, countvd))
diametervd = np.vstack((diametervd, accuracy_vd))
diameterhd = np.vstack((counthdo, counthd))
diameterhd = np.vstack((diameterhd, accuracy_hd))

print(diameterhd.shape)


'''
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(counto)
ax1.plot(countp2)
ax1.axis([0,len(counto),-1000,max(counto)*1.07])
#ax1.plot(countp2)
ax1.set_ylabel('pixelcount')

ax2 = ax1.twinx()
ax2.plot(iou,'r-')
ax2.axis([0,len(iou),0.3,3])
ax2.set_ylabel('IOU',color='r')
plt.savefig('testresult.png')
plt.gcf().clear()

plt.plot(counto)
plt.plot(countp)
plt.plot(countp2)
plt.savefig('markingresult.png')
plt.gcf().clear()
plt.plot(iou)
plt.savefig('iou.png')
plt.show()
plt.gcf().clear()
'''

print(np.average(iou))

np.save('markresult.npy', countp)
np.save('iou.npy', iou)
    
plotresults(counto, countp2, iou, diametervd, diameterhd, 0, testcount[0], 'EP')
plotresults(counto, countp2, iou, diametervd, diameterhd, testcount[0], testcount[0] + testcount[1] ,'larva')
plotresults(counto, countp2, iou, diametervd, diameterhd, testcount[0] + testcount[1], testcount[0] + testcount[1] + testcount[2],'Adult')



