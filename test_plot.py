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
from parameters import Parameters
from skimage import morphology


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return ( intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def generatefolders(name):
    directory = "/media/zlab-1/Data/Lian/keras/nData"
    directory2 = "./AD"
    #EP01 : 18-19
    start1 = parameters.teststartEP
    end1 = parameters.testendEP
    #Larva02 : 53 - 54
    start2 = parameters.teststartLA
    end2 = parameters.testendLA
    start3 = parameters.teststartAD
    end3 = parameters.testendAD
    folders=sorted(glob(directory+"/*/"))+sorted(glob(directory2+"/*/"))
    #folder1=folders[start1:end1]+folders[start2:end2] + folders[start3:end3]

    if name == 'EP':
        folder1 = folders[start1:end1]
    elif name == 'larva':
        folder1 = folders[start2:end2]
    else:
        folder1 = folders[start3:end3]

    return folder1


def prepare_data_test(name):


    train=[]
    y=[]

    
    folder1 = generatefolders(name)

    X_test=[]
    y_test=[]
    #testcount=[]

    for i in folder1:
        print(i)
        file=sorted(glob(i+'*.png'))
        print("number of files is ", len(file) / 2)
        #testcount.append(len(file) / 2)

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

    #print(testcount)
    return (X_test, y_test)

def analyzetest(a, name):
    print('Total Number for'+ name + ':',len(a))
    
    counto, countp2, iou, diametervd, diameterhd = calculatearea(a, name)

    return counto, countp2, iou, diametervd, diameterhd
    
    '''    
    plotresults(counto, countp2, iou, diametervd, diameterhd, 0, testcount[0], 'EP')
    plotresults(counto, countp2, iou, diametervd, diameterhd, testcount[0], testcount[0] + testcount[1] ,'larva')
    plotresults(counto, countp2, iou, diametervd, diameterhd, testcount[0] + testcount[1], testcount[0] + testcount[1] + testcount[2],'Adult')
    '''


def calculatearea(a, name):
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
        cv2.imwrite('/media/zlab-1/Data/Lian/keras/result/'+name+'{}.jpg'.format(format(i,'05')),dst)
        cv2.imwrite('/media/zlab-1/Data/Lian/keras/result/'+name+'{}_original.jpg'.format(format(i,'05')),X_test[i])
        
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
        if name == 'larva':
            c1 = morphology.remove_small_objects(c1, 350)
        #np.save('./bboxtry/{}.npy'.format(i),c1)

        count3 = ((150<c1)&(c1<260)).sum()
        #count3 = (c_mor == True).sum()
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

    print(np.average(iou))
    np.save('markresult.npy', countp)
    np.save('iou.npy', iou)

    return(counto, countp2, iou, diametervd, diameterhd)

#Mode / View / Control
#Start with program

parameters = Parameters()

K.set_image_data_format('channels_last') 

smooth=1.
#model = load_model('/media/zlab-1/Data/Lian/keras/EP/weights1.h5', custom_objects={'dice_coef_loss':dice_coef,'dice_coef':dice_coef})
model = load_model(parameters.directory+'/weights1.h5', custom_objects={'dice_coef_loss':dice_coef,'dice_coef':dice_coef})

for name in ['EP', 'larva', 'AD']:

    X_test, y_test = prepare_data_test(name)

    a = model.predict(X_test, batch_size=32, verbose=2)

    counto, countp2, iou, diametervd, diameterhd = calculatearea(a, name)

    plotresults(counto, countp2, iou, diametervd, diameterhd, name)

