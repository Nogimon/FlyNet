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
from BilinearUpSampling import *


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return ( intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

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
    #train, y = generatetrain(foldertrain)
    #X_validate, y_validate = generatevalidate(foldervalidate)
    X_test, y_test = generatetest(foldertest)

    train, y = [], []
    X_validate, y_validate = [], []






    
    return (train, y, X_validate, y_validate, X_test, y_test)

def generatetest(foldertest):
    imgsize = 320
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
        
        '''
        dst = cv2.addWeighted(cc,0.5,bb,0.5,0)    
        cv2.imwrite('/media/zlab-1/Data/Lian/keras/result0529/'+name+'{}.jpg'.format(format(i,'05')),dst)
        cv2.imwrite('/media/zlab-1/Data/Lian/keras/result0529/'+name+'{}_original.jpg'.format(format(i,'05')),X_test[i])
        '''


        #Create images in preparation for flaw images looks like fig2
        cv2.imwrite('/media/zlab-1/Data/Lian/keras/resultfcn/'+name+'{}_gt.jpg'.format(format(i,'05')),bb)    
        cv2.imwrite('/media/zlab-1/Data/Lian/keras/resultfcn/'+name+'{}_pd.jpg'.format(format(i,'05')),cc)
        cv2.imwrite('/media/zlab-1/Data/Lian/keras/resultfcn/'+name+'{}_original.jpg'.format(format(i,'05')),X_test[i])
        '''
        new_im = Image.new('RGB', (256,256))
        new_im.paste(X_test[i], (0,0))
        new_im.paste(cc, (0,128))
        new_im.save('/media/zlab-1/Data/Lian/keras/result0529sep/'+name+'{}_original.jpg'.format(format(i,'05')))
        new_im2 = Image.new('RGB', (256,256))
        new_im2.paste(X_test[i], (0,0))
        new_im2.paste(bb, (128,0))
        new_im2.save('/media/zlab-1/Data/Lian/keras/result0529sep/'+name+'{}_original.jpg'.format(format(i,'05')))
        '''
        
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
    counto = np.asarray(counto)
    countp2 = np.asarray(countp2)

    #accuracy_vd = 1 - (countvd - countvdo) / (countvdo.astype(float))
    accuracy_vd = (countvd) / (countvdo.astype(float))
    accuracy_hd = (counthd) / (counthdo.astype(float))

    print(counthdo.shape, counthd.shape, accuracy_hd.shape)


    diametervd = np.vstack((countvdo, countvd))
    diametervd = np.vstack((diametervd, accuracy_vd))
    diameterhd = np.vstack((counthdo, counthd))
    diameterhd = np.vstack((diameterhd, accuracy_hd))

    #parameters = Parameters()
    counto = counto * parameters.yfactor * parameters.xfactor
    countp2 = countp2 * parameters.yfactor * parameters.xfactor
    diametervd = diametervd * parameters.yfactor
    diameterhd = diameterhd * parameters.xfactor

    print(np.average(iou))

    return(counto, countp2, iou, diametervd, diameterhd)

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


#Mode / View / Control
#Start with program
if __name__ == '__main__':
    parameters = Parameters()
    directory = "./nTrain"

    K.set_image_data_format('channels_last') 

    smooth=1.
    model = load_model(directory+'/weights_fcn_new0.h5', custom_objects={'BilinearUpSampling2D':BilinearUpSampling2D, 'dice_coef_loss':dice_coef,'dice_coef':dice_coef})
    
    #model = load_model(directory+'/weights_fcn' + str(parameters.testfolder) + '.h5', custom_objects={'BilinearUpSampling2D':BilinearUpSampling2D,'softmax_sparse_crossentropy_ignoring_last_label':softmax_sparse_crossentropy_ignoring_last_label, 'sparse_accuracy_ignoring_last_label':sparse_accuracy_ignoring_last_label})
    

    train, y, X_validate, y_validate, X_test, y_test = generateData()


    a = model.predict(X_test, batch_size=8, verbose=2)

    counto, countp2, iou, diametervd, diameterhd = calculatearea(a, 'Larva')

    #for larva maybe skip?
    
    name = 'pupa'
    if name == 'larva':
        plotresults(counto, countp2, iou, diameterhd, name + '_' + str(parameters.testfolder))
    else:
        plotresults(counto, countp2, iou, diametervd, name + '_' + str(parameters.testfolder))

