# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 11:57:34 2017

@author: L
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
import matplotlib.pyplot as plt

def prepare_data(target, train, y, X_test, y_test):

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
        end1 = 20
        start2 = 81
        end2 = 83
        start3 = -14
        end3 = -12
        folders=sorted(glob(directory+"/*/"))+sorted(glob(directory2+"/*/"))
        folder1=folders[start1:end1]+folders[start2:end2] + folders[start3:end3]
        folder=folders[0:start1]+folders[end1:start2]+folders[end2:start3] + folders[end3:]

    shift = 10

    
    for i in folder:
        print(i)
        fileo = sorted(glob(i+'*[0-9].png'))
        filem = sorted(glob(i+'*mask.png'))
        print(len(fileo))
        print(len(filem))
        k = 0
        for j in fileo:
            name=os.path.basename(j)
            img=np.asarray(Image.open(j))
            #img = cv2.cvtColor(cv2.imread(j), cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(img, (128, 128), cv2.INTER_LINEAR)
            
            if (k%7==0):
                augmented = transform.rotate(resized, 90)
            elif (k%7==1):
                augmented = transform.rotate(resized, 180)
            elif (k%7==2):
                augmented = transform.rotate(resized, 270)
            elif (k%7==3):
                augmented = np.roll(resized, shift, axis = 1)
                augmented[:,0:shift] = 0
            elif (k%7==4):
                augmented = np.roll(resized, shift, axis = 0)
                augmented[0:shift,:] = 0
            elif (k%7==5):
                augmented = np.roll(resized, -shift, axis = 1)
                augmented[:,-shift:] = 0
            elif (k%7==6):
                augmented = np.roll(resized, -shift, axis = 0)
                augmented[-shift:,:] = 0
            

            train.append(resized)
            train.append(augmented)
            
            k+=1
        k = 0
        for j in filem:
            name=os.path.basename(j)
            img=np.asarray(Image.open(j))
            #img = cv2.cvtColor(cv2.imread(j), cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(img, (128, 128), cv2.INTER_LINEAR)
            
            if (k%7==0):
                augmented = transform.rotate(resized, 90)
            elif (k%7==1):
                augmented = transform.rotate(resized, 180)
            elif (k%7==2):
                augmented = transform.rotate(resized, 270)
            elif (k%7==3):
                augmented = np.roll(resized, shift, axis = 1)
                augmented[:,0:shift] = 0
            elif (k%7==4):
                augmented = np.roll(resized, shift, axis = 0)
                augmented[0:shift,:] = 0
            elif (k%7==5):
                augmented = np.roll(resized, -shift, axis = 1)
                augmented[:,-shift:] = 0
            elif (k%7==6):
                augmented = np.roll(resized, -shift, axis = 0)
                augmented[-shift:,:] = 0
            


            
            k+=1
            
            y.append(resized)
            y.append(augmented)
        
    
    
    train=np.asarray(train)
    y=np.asarray(y)
    train=train[...,np.newaxis]
    y=y[...,np.newaxis]
    

    for i in folder1:
        print(i)
        file=glob(i+'*.png')
        for j in file:
            name=os.path.basename(j)
            img=np.asarray(Image.open(j))
            #img = cv2.cvtColor(cv2.imread(j), cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(img, (128, 128), cv2.INTER_LINEAR)
    
            if name.replace('.png','').isdigit() == True:
                X_test.append(resized)
            if name.replace('.png','').isdigit() == False:
                y_test.append(resized)
    
    X_test=np.asarray(X_test)
    y_test=np.asarray(y_test)
    X_test=X_test[...,np.newaxis]
    y_test=y_test[...,np.newaxis]
    
    return train, y, X_test, y_test


def plotresults(p_ground, p_count,p_iou,p_diameter,start,end,name):
    p_ground = p_ground[start:end]
    p_count = p_count[start:end]
    p_iou = p_iou[start:end]

    plt.plot(p_ground)
    plt.plot(p_count)
    plt.savefig('./resultimage/pixelcount_'+name+'.png')
    plt.gcf().clear()

    plt.plot(p_iou)
    plt.savefig('./resultimage/iou_'+name+'.png')
    plt.gcf().clear()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(p_ground)
    ax1.plot(p_count)
    ax1.axis([0,len(p_count),-1000,max(p_count)*1.07])
    #ax1.plot(countp2)
    ax1.set_ylabel('pixelcount')

    ax2 = ax1.twinx()
    ax2.plot(p_iou,'r-')
    ax2.axis([0,len(p_iou),0.3,3])
    ax2.set_ylabel('IOU',color='r')
    plt.savefig('./resultimage/testresult_'+name+'.png')
    plt.gcf().clear()

    return