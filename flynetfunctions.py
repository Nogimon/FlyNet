from PIL import Image
import numpy as np
import os
from glob import glob
import cv2
import time
'''
from keras.models import Model, load_model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
'''
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.axis import XAxis
from skimage import transform
from parameters import Parameters
from scipy.signal import find_peaks_cwt

def prepare_data(target, train, y, X_test, y_test):

    parameters = Parameters()
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
        start1 = parameters.trainstartEP
        end1 = parameters.trainendEP
        start2 = parameters.trainstartLA
        end2 = parameters.trainendLA
        start3 = parameters.trainstartAD
        end3 = parameters.trainendAD
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

def plotresults(p_ground, p_count, p_iou, diametervd, name):
    xaxis = np.array(range(len(p_ground)))
    parameters=Parameters()
    xaxis = xaxis/parameters.timefactor-1

    #p_ground = p_ground[start:end]
    #p_count = p_count[start:end]
    #p_iou = p_iou[start:end]

    #diameterhd = diameterhd[:,start:end]
    #diametervd = diametervd[:,start:end]


    #plot individual pixel count and iou
    '''
    
    plt.figure(num = None, figsize = (8, 6), dpi = 200)

    plt.plot(xaxis,p_ground)#, color = 'darkgreen')
    plt.plot(xaxis,p_count)#, color = 'gold')
    #plt.savefig('./resultimage/pixelcount_'+time.asctime(time.localtime(time.time()))+name+'.png')
    plt.savefig('./resultimage/pixelcount_'+name +'.png')
    plt.clf()
    #plt.gcf().clear()

    plt.figure(num = None, figsize = (8, 2), dpi = 200)
    plt.ylim(0, 1)
    plt.plot(xaxis,p_iou, color = '#8c564b')
    #plt.savefig('./resultimage/iou_'+time.asctime(time.localtime(time.time()))+name+'.png')
    plt.savefig('./resultimage/iou_'+name+'.png')
    plt.clf()
    '''


    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(p_ground, color = '#006064')
    ax1.plot(p_count, color = '#F57C00')
    ax1.axis([0,len(p_count),-1000,max(p_ground)*1.07])
    #ax1.plot(countp2)
    ax1.set_ylabel('pixelcount')

    ax2 = ax1.twinx()
    ax2.plot(p_iou)#, color = '#006064') #color = '#004D40')
    ax2.axis([0,len(p_iou),0.3,3])
    ax2.set_ylabel('IOU',color='#7f7f7f')
    plt.savefig('./resultimage/testresult_'+name+'.png')
    plt.clf()
    '''
    
    plt.figure()
    
    gs = gridspec.GridSpec(5, 4)
    gs.update(hspace = 0.2)
    plt.subplot(gs[0 : 2, :])
    #XAxis.set_ticks_position(top)
    plt.tick_params(direction = 'in')
    l1=plt.plot(xaxis,p_ground, label = 'GroudTruth')#, color = '#006064')
    l2=plt.plot(xaxis,p_count, label = 'ModelPrediction')#, color = '#F57C00')
    plt.title('Ground Truth & Prediction vs Time', fontsize=18,va='bottom',ha='center')
    

    #plt.xlabel('Time(s)')
    plt.ylabel('Area('+r'$\mu{m}^2$'+')', fontsize=14,va='center',ha='center')
    #plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.xlim(0,xaxis[-1])
    plt.gca().axes.xaxis.set_ticklabels([])
    #plt.axis([0,len(p_count),-1000,max(p_ground)*1.07])
    #ax1.plot(countp2)
    #ax1.set_ylabel('pixelcount')

    
    #ax2.set_ylabel('IOU',color='#7f7f7f')
    #plt.title(name + 'Heart Area Change')
    plt.subplot(gs[2: 4, :])

    #if (name == 'larva'):
    #    plt.plot(xaxis,diameterhd[0], label = 'GroudTruth')#, color = '#006064')
    #    plt.plot(xaxis,diameterhd[1], label = 'ModelPrediction')#, color = '#F57C00')
    #savenpy(diameterhd[1], name)
    
    plt.plot(xaxis,diametervd[0], label = 'GroudTruth')#, color = '#006064')
    plt.plot(xaxis,diametervd[1], label = 'ModelPrediction')#, color = '#F57C00')
    #savenpy(diametervd[1], name)
    #plt.title('ground truth & prediction Vs Time')
    #plt.xlabel('Time(s)')
    
    plt.ylabel('Diameter(um)', fontsize=14,va='bottom',ha='center')
    plt.xlim(0,xaxis[-1])
    plt.gca().axes.xaxis.set_ticklabels([])


    plt.subplot(gs[4, :])
    l3=plt.plot(xaxis,p_iou, color = '#1B5E20')#33691E')# color = '#006064')
    #plt.title('iou Vs Time')
    
    plt.xlabel('Time(s)', fontsize=14)
    plt.ylabel('Iou', fontsize=14,va='bottom',ha='center')
    plt.xlim(0,xaxis[-1])
    plt.ylim(0, 1)
    #plt.legend()
    plt.figlegend((l1,l2,l3),('GroudTruth','ModelPrediction','accuracy'),'upper right')
    #plt.savefig('./resultimage/arearesult_'+time.asctime(time.localtime(time.time()))+name+'.png')
    plt.savefig('./resultimage/arearesult_'+name+'.png')
    plt.clf()
    '''


    #savenpy(diameter)

    
    plt.figure()
    plt.plot(xaxis,p_iou, label = 'GroudTruth')#, color = '#006064')
    plt.plot(xaxis, p_iou, label = 'ModelPrediction')#, color = '#F57C00')
    plt.plot(xaxis, p_iou, label = "IOU", color = '#1B5E20')
    plt.legend()
    plt.savefig('./resultimage/arearesult_'+name+'legend.png')
    plt.clf()
    '''

    '''
    plt.figure()
    
    gs = gridspec.GridSpec(2, 1, height_ratios = [3, 1])
    gs.update(hspace = 0)
    plt.subplot(gs[0])
    #XAxis.set_ticks_position(top)
    plt.tick_params(direction = 'in')
    plt.plot(diametervd[0], label = 'GroudTruth')#, color = '#006064')
    plt.plot(diametervd[1], label = 'ModelPrediction')#, color = '#F57C00')
    #plt.axis([0,len(p_count),-1000,max(p_ground)*1.07])
    #ax1.plot(countp2)
    #ax1.set_ylabel('pixelcount')

    plt.subplot(gs[1])
    plt.plot(diametervd[2], color = '#1B5E20') #color = '#004D40')
    plt.ylim(0, 2)
    #ax2.set_ylabel('IOU',color='#7f7f7f')
    #plt.title(name + 'Heart Diameter Change')
    plt.savefig('./resultimage/diamterresult_'+name+'.png')
    plt.clf()

    
    plt.figure(num = None, figsize = (8, 6), dpi = 200)
    plt.plot(diametervd[0])#, color = 'g')
    plt.plot(diametervd[1])#, color = '#ff7f0e')
    plt.savefig('./resultimage/vertical_diameter_'+name+'.png')
    plt.clf()

    plt.figure(num = None, figsize = (8, 2), dpi = 200)
    plt.ylim(0, 2)
    plt.plot(diametervd[2], color = '#9467bd')
    plt.savefig('./resultimage/vertical_diameter_accuracy_'+name+'.png')
    plt.clf()
    '''
    '''
    plt.figure(num = None, figsize = (8, 6), dpi = 200)
    plt.plot(diameterhd[0])
    plt.plot(diameterhd[1])
    plt.savefig('./resultimage/horizontal_diameter_'+name+'.png')
    plt.clf()

    plt.figure(num = None, figsize = (8, 2), dpi = 200)
    plt.ylim(0, 2)
    plt.plot(diameterhd[2], color = 'darkblue')
    plt.savefig('./resultimage/horizontal_diameter_accuracy_'+name+'.png')
    plt.close('all')
    '''


    return



def getbox(image):
    one = np.where(image > 150)
    vertdiameter = np.max(one[0]) - np.min(one[0])
    horidiameter = np.max(one[1]) - np.min(one[1])
    i = 0
    if (horidiameter > 80):
        plt.imshow(image)
        plt.savefig("./test.png")
        plt.close('all')
    return (vertdiameter, horidiameter)


def savenpy(data, savename):
    np.save('./resultimage/'+savename+'.npy', data)


def findpeaks(data, step):
    #Find peaks
    peak = []
    peaks = []
    peaks = find_peaks_cwt(data, np.arange(1, step))
    peaks2 = find_peaks_cwt(-data, np.arange(1, step))

    if len(peaks) == 0:
        print("peaks not found")
        peaks = np.zeros(1)
    if len(peaks2) == 0:
        print("peaks2 not found")
        peaks2 = np.zeros(1)
    peaks = peaks.astype(int)
    peaks2 = peaks2.astype(int)
    #peaks = np.sort(np.asarray(peaks+peaks2))
    high = (np.vstack((peaks, np.asarray(data[peaks]))))
    low = (np.vstack((peaks2, np.asarray(data[peaks2]))))

    
    '''
    #plot the peak finding result
    #plt.figure()
    plt.plot(data, color = 'r')
    
    #height = np.flip(np.rot90(height), axis = 0)
    plt.scatter(high[0], high[1])
    plt.scatter(low[0], low[1])
    plt.show()
    plt.gcf().clear()
    '''


    #record height data
    high_ave = np.average(high[1])
    high_std = np.std(high[1])
    low_ave = np.average(low[1])
    low_std = np.std(low[1])
    #calculate heart rate
    timerange = (peaks[-1] - peaks[0]) / 129.0
    if timerange == 0:
        heartrate = 0
    else:
        heartrate = len(peaks) / float(timerange)
    return (heartrate, high_ave, high_std, low_ave, low_std)