
from datetime import datetime
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
from matplotlib import gridspec
from matplotlib.axis import XAxis
from skimage import transform
from parameters import Parameters


def generatefolders(name):
    directory = "/media/zlab-1/Data/Lian/keras/nTrain/" + name
    
    folders=sorted(glob(directory+"/*/"))
    return folders

def prepare_data(name):

    folders = generatefolders(name)
           
    
    foldertrain = folders[:]    
    foldervalidate = folders[parameters.valifolder:parameters.valifolder+1]
    foldertest = folders[parameters.testfolder:parameters.testfolder+1]
    #foldertrain = folders - foldertest - foldervalidate
    del foldertrain[parameters.valifolder]
    del foldertrain[parameters.testfolder]

    train, y = generatetrain(foldertrain)
    X_validate, y_validate = generatevalidate(foldervalidate)
    X_test, y_test = generatetest(foldertest)
    
    return (train, y, X_validate, y_validate, X_test, y_test)

def generatetrain(foldertrain):
    train=[]
    y=[]
    #generate train data
    shift = 10
    for i in foldertrain:
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

    return (train, y)

def generatevalidate(foldervalidate):
    X_validate=[]
    y_validate=[]
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

    return (X_validate, y_validate)

def generatetest(foldertest):
    X_test = []
    y_test = []
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

    return (X_test, y_test)


def generateData():

    name = "Larva"
    train, y, X_validate, y_validate, X_test, y_test = prepare_data(name)
    '''
    train=[]
    y=[]
    X_validate=[]
    y_validate=[]
    X_test = []
    y_test = []
    '''
    for name in ["EP", "AD"]:
        train1, y1, X_validate1, y_validate1, X_test1, y_test1 = prepare_data(name)
        train = np.vstack((train, train1))
        y = np.vstack((y,y1))
        X_validate= np.vstack((X_validate, X_validate1))
        y_validate= np.vstack((y_validate,y_validate1))
        X_test= np.vstack((X_test, X_test1))
        y_test= np.vstack((y_test, y_test1))

    
    train=np.asarray(train)
    y=np.asarray(y)
    X_validate=np.asarray(X_validate)
    y_validate = np.asarray(y_validate)
    X_test=np.asarray(X_test)
    y_test=np.asarray(y_test)
    train=train[...,np.newaxis]
    y=y[...,np.newaxis]
    X_test=X_test[...,np.newaxis]
    y_test=y_test[...,np.newaxis]

    X_validate=X_validate[...,np.newaxis]
    y_validate=y_validate[...,np.newaxis]

    return (train, y, X_validate, y_validate, X_test, y_test)



def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

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


if __name__ == '__main__':

    directory = "./nTrain"

    #get the data
    parameters = Parameters()
    train, y, X_validate, y_validate, X_test, y_test = generateData()


    #Set the model
    K.set_image_data_format('channels_last') 
    smooth=1.
    model=get_model()
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    #model_checkpoint = ModelCheckpoint(directory+'/weights1-{}-.h5'.format(str(datetime.now())), monitor='val_loss', save_best_only=True)
    model_checkpoint = ModelCheckpoint(directory+'/weights1_test' + str(parameters.testfolder) + '.h5', monitor='val_loss', save_best_only=True)
    earlystop = EarlyStopping(monitor='val_loss', patience=5, mode='auto')
    
    

    #Train
    model.fit(train, y, batch_size=32, epochs=150, verbose=1, shuffle=True, callbacks=[model_checkpoint, earlystop],validation_data=(X_test, y_test))

    #Test
    #a=model.predict(X_test, batch_size=32, verbose=2)









'''

    history=model.fit(x, y, 
        batch_size=32, 
        epochs=150, 
        verbose=1, 
        shuffle=True,
        callbacks=[ModelCheckpoint(directory+'/weights1-{}-.h5'.format(str(datetime.now()),history[-1,1]), monitor='val_loss', save_best_only=True), 
        EarlyStopping(monitor='val_loss', patience=5, mode='auto')],
        validation_data=(x_val[_], y_val[_]))
'''


