
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
from keras.regularizers import l2

from skimage import transform
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.axis import XAxis
from skimage import transform
from parameters import Parameters
from BilinearUpSampling import *
from DataGeneratorForFCN import DataGenerator


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
    '''
    foldertrain = folders[8:10]
    foldervalidate = folders[9:10]
    foldertest = folders[9:10]
    '''
    train, y = generatetrain(foldertrain)
    X_validate, y_validate = generatevalidate(foldervalidate)
    X_test, y_test = generatetest(foldertest)






    
    return (train, y, X_validate, y_validate, X_test, y_test)


def generatetrain(foldertrain):
    imgsize = 320
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
            resized = cv2.resize(img, (imgsize, imgsize), cv2.INTER_LINEAR)
            
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
            resized = cv2.resize(img, (imgsize, imgsize), cv2.INTER_LINEAR)


            #resized = resized * 255
            
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
    imgsize = 320
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
            resized = cv2.resize(img, (imgsize, imgsize), cv2.INTER_LINEAR)
            #resized = img

            if name.replace('.png','').isdigit() == True:
                X_validate.append(resized)
            if name.replace('.png','').isdigit() == False:
                #resized = resized * 255
                y_validate.append(resized)

    X_validate=np.asarray(X_validate)
    y_validate = np.asarray(y_validate)

    return (X_validate, y_validate)

def generatetest(foldertest):
    X_test = []
    y_test = []
    imgsize = 320
    #generate test data
    for i in foldertest:
        print("the test data is", i)
        file=glob(i+'*.png')
        for j in file:
            name=os.path.basename(j)
            img=np.asarray(Image.open(j))
            #img = cv2.cvtColor(cv2.imread(j), cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(img, (imgsize, imgsize), cv2.INTER_LINEAR)
    
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

    '''
    for name in ["EP", "AD"]:
        train1, y1, X_validate1, y_validate1, X_test1, y_test1 = prepare_data(name)
        train = np.vstack((train, train1))
        y = np.vstack((y,y1))
        X_validate= np.vstack((X_validate, X_validate1))
        y_validate= np.vstack((y_validate,y_validate1))
        X_test= np.vstack((X_test, X_test1))
        y_test= np.vstack((y_test, y_test1))
    '''
    #np.save("./train.npy", train)
    #np.save("./y.npy", y)
    #np.save("./X_validate.npy", X_validate)
    #np.save("./y_validate.npy", y_validate)

    '''
    # Add new data
    foldernew = "/media/zlab-1/Data/Lian/keras/nTrain/newfly/"
    train1, y1 = generatetrainnew(foldernew)

    train = np.vstack((train, train1))
    y = np.vstack((y, y1))
    '''

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


def generateDataSimple(colorgt):
    imgsize = 320
    directory = "/media/zlab-1/Data/Lian/keras/nTrain/Larva"
    folders=sorted(glob(directory+"/*/"))
    
    foldertrain = folders[8:10]
    foldervalidate = folders[9:10]

    train=[]
    y=[]
    #generate train data
    for i in foldertrain:
        print(i)
        fileo = sorted(glob(i+'*[0-9].png'))
        filem = sorted(glob(i+'*mask.png'))
        for j in fileo:
            name=os.path.basename(j)
            img=np.asarray(Image.open(j))
            #img = cv2.cvtColor(cv2.imread(j), cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(img, (imgsize, imgsize), cv2.INTER_LINEAR)
            train.append(resized)
        for j in filem:
            name=os.path.basename(j)
            img=np.asarray(Image.open(j))
            #img = cv2.cvtColor(cv2.imread(j), cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(img, (imgsize, imgsize), cv2.INTER_LINEAR)
            if (colorgt == True):
                resized = resized * 255
            
            y.append(resized)
    train=np.asarray(train)
    y=np.asarray(y)

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
            resized = cv2.resize(img, (imgsize, imgsize), cv2.INTER_LINEAR)
            #resized = img

            if name.replace('.png','').isdigit() == True:
                X_validate.append(resized)
            if name.replace('.png','').isdigit() == False:
                if (colorgt == True):
                    resized = resized * 255
                y_validate.append(resized)

    X_validate=np.asarray(X_validate)
    y_validate = np.asarray(y_validate)


    train=train[...,np.newaxis]
    y=y[...,np.newaxis]
    X_validate=X_validate[...,np.newaxis]
    y_validate=y_validate[...,np.newaxis]
    '''
    if (colorgt == True):
        y = np.append(y, y, axis = 3)
        y_validate = np.append(y_validate, y_validate, axis = 3)
    '''
    print(y.shape)
    print(train.shape)
    return (train, y, X_validate, y_validate)


def evaluateTest(testresult):
    print (testresult.shape)
    return



def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    #y_true_f /= 255.0
    y_pred_f = K.flatten(y_pred)
    #y_pred_f /= 255.0
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def softmax_sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean

def sparse_accuracy_ignoring_last_label(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)),
                       nb_classes + 1)
    unpacked = tf.unstack(y_true, axis=-1)
    legal_labels = ~tf.cast(unpacked[-1], tf.bool)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    return K.sum(tf.to_float(legal_labels & K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))) / K.sum(tf.to_float(legal_labels))

def get_model_fcn():
    weight_decay = 0
    inputs = Input((320, 320, 1))
    conv1 = Conv2D(64, (3, 3), activation = 'relu', border_mode = 'same', W_regularizer=l2(weight_decay))(inputs)
    conv1 = Conv2D(64, (3, 3), activation = 'relu', border_mode = 'same', W_regularizer=l2(weight_decay))(conv1)
    pool1 = MaxPooling2D(pool_size = (2,2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation = 'relu', border_mode = 'same', W_regularizer=l2(weight_decay))(pool1)
    conv2 = Conv2D(128, (3, 3), activation = 'relu', border_mode = 'same', W_regularizer=l2(weight_decay))(conv2)
    pool2 = MaxPooling2D(pool_size = (2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation = 'relu', border_mode = 'same', W_regularizer=l2(weight_decay))(pool2)
    conv3 = Conv2D(256, (3, 3), activation = 'relu', border_mode = 'same', W_regularizer=l2(weight_decay))(conv3)
    conv3 = Conv2D(256, (3, 3), activation = 'relu', border_mode = 'same', W_regularizer=l2(weight_decay))(conv3)
    pool3 = MaxPooling2D(pool_size = (2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation = 'relu', border_mode = 'same', W_regularizer=l2(weight_decay))(pool3)
    conv4 = Conv2D(512, (3, 3), activation = 'relu', border_mode = 'same', W_regularizer=l2(weight_decay))(conv4)
    conv4 = Conv2D(512, (3, 3), activation = 'relu', border_mode = 'same', W_regularizer=l2(weight_decay))(conv4)
    pool4 = MaxPooling2D(pool_size = (2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation = 'relu', border_mode = 'same', W_regularizer=l2(weight_decay))(pool4)
    conv5 = Conv2D(512, (3, 3), activation = 'relu', border_mode = 'same', W_regularizer=l2(weight_decay))(conv5)
    conv5 = Conv2D(512, (3, 3), activation = 'relu', border_mode = 'same', W_regularizer=l2(weight_decay))(conv5)
    pool5 = MaxPooling2D(pool_size = (2, 2))(conv5)

    conv6 = Conv2D(4096, (7, 7), activation = 'relu', border_mode = 'same', W_regularizer=l2(weight_decay))(pool5)
    drop1 = Dropout(0.5)(conv6)
    conv6 = Conv2D(4096, (1, 1), activation = 'relu', border_mode = 'same', W_regularizer=l2(weight_decay))(drop1)
    drop2 = Dropout(0.5)(conv6)

    #upconv1 = Conv2DTranspose(256, (3, 3), strides=(7, 7), padding='same')(drop2)

    conv7 = Conv2D(1, (1, 1), kernel_initializer = 'he_normal', activation = 'sigmoid', padding = 'valid', strides = (1, 1), W_regularizer=l2(weight_decay))(drop2)
    up1 = BilinearUpSampling2D(size = (32, 32))(conv7)

    #conv8 = Conv2D(1, (1, 1), activation = 'sigmoid', padding = 'same')(up1)

    model = Model(inputs = [inputs], outputs = [up1])
    return model





if __name__ == '__main__':

    directory = "./nTrain"
    #True if the image last channel max is 255
    colorgt = False
    #True if use generator
    augflag = True
    #get the data
    parameters = Parameters()

    if (augflag == False):
        train, y, X_validate, y_validate, X_test, y_test = generateData()
        #train, y, X_validate, y_validate = generateDataSimple(colorgt)
        np.save("./X_validate320.npy", X_validate)
        np.save("./y_validate329.npy", y_validate)
    else:
        trainGenerator = DataGenerator().dataGenerateWithAug("./nTrain")
        X_validate = np.load("./X_validate.npy")
        y_validate = np.load("./y_validate.npy")
        X_validate=X_validate[...,np.newaxis]
        y_validate=y_validate[...,np.newaxis]




    #Set the model
    K.set_image_data_format('channels_last') 
    smooth=1.
    model=get_model_fcn()
    #model.compile(optimizer=Adam(lr=1e-5), loss=softmax_sparse_crossentropy_ignoring_last_label, metrics=[sparse_accuracy_ignoring_last_label])
    
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    model.summary()
    #model_checkpoint = ModelCheckpoint(directory+'/weights1-{}-.h5'.format(str(datetime.now())), monitor='val_loss', save_best_only=True)
    model_checkpoint = ModelCheckpoint(directory+'/weights_fcn_new' + str(parameters.testfolder) + '.h5', monitor='val_loss', save_best_only=True)
    earlystop = EarlyStopping(monitor='val_loss', patience=5, mode='auto')
    
    

    #Train
    if (augflag == False):
        history = model.fit(train, y, batch_size=2, epochs=20, verbose=1, shuffle=True, callbacks=[model_checkpoint, earlystop],validation_data=(X_validate, y_validate))
        nphistory = np.array(history.history['dice_coef'])
        np.savetxt("./fcnhistory_train.txt", nphistory, delimiter = ",")
        nphistoryv = np.array(history.history['val_dice_coef'])
        np.savetxt("./fcnhistory_val.txt", nphistoryv, delimiter = ",")
    else:
        history = model.fit_generator(generator = trainGenerator, steps_per_epoch = 25000, epochs = 20, verbose = 1)#, callbacks = [model_checkpoint, earlystop], validation_data = (X_validate, y_validate))
        nphistory = np.array(history.history['dice_coef'])
        np.savetxt("./fcnhistory_train.txt", nphistory, delimiter = ",")
        nphistoryv = np.array(history.history['val_dice_coef'])
        np.savetxt("./fcnhistory_val.txt", nphistoryv, delimiter = ",")


    #Test
    testresult=model.predict(X_validate, batch_size=8, verbose=2)
    evaluateTest(testresult)









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


