#Fine tune for purepredict model

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
from skimage import transform
from flynetfunctions import prepare_data

directory = "/media/zlab-1/Data/Lian/keras/nData"

smooth=1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def prepare_data(name):

    folders = "/media/zlab-1/Data/Lian/keras/Purepredict/purefinetune/"#generatefolders(name)
    train=[]
    y=[]
    X_validate=[]
    y_validate=[]
    X_test = []
    y_test = []
    
    foldertrain = [folders]

    #generate train data
    shift = 10
    for i in foldertrain:
        print(i)
        fileo = sorted(glob(i+'*original.jpg'))
        filem = sorted(glob(i+'*[0-9].jpg'))
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
            img2=np.asarray(Image.open(j))
            #convert to 0 and 1
            #extract blue channel
            img2 = img2[:,:,2]
            img = img2 > 0
            img = img.astype(int)

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
    
    
    '''
    #generate validate data
    for i in foldervalidate:
        print("the validate data is", i)
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
                X_validate.append(resized)
            if name.replace('.png','').isdigit() == False:
                y_validate.append(resized)

    X_validate=np.asarray(X_validate)
    y_validate = np.asarray(y_validate)
    
    #generate test data
    for i in foldertest:
        print("the test data is", i)
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
    
	'''


    return (train, y)#, X_validate, y_validate, X_test, y_test)



train, y = prepare_data('')

train=np.asarray(train)
y=np.asarray(y)

train=train[...,np.newaxis]
y=y[...,np.newaxis]

model = load_model(directory+'/weights1.h5', custom_objects={'dice_coef_loss':dice_coef,'dice_coef':dice_coef})

model_checkpoint = ModelCheckpoint('/media/zlab-1/Data/Lian/keras/Purepredict/newweights.h5'.format(), monitor='val_loss', save_best_only=True)


model.fit(train, y, batch_size=32, epochs=1, verbose=1, shuffle=True, callbacks=[model_checkpoint])#,validation_data=(X_test, y_test))

model.save("/media/zlab-1/Data/Lian/keras/Purepredict/newweights.h5")