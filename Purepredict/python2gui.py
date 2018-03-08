#LD

from Tkinter import *
import tkFileDialog

import numpy as np
from skimage import io
from skimage import morphology
from matplotlib import pyplot as plt
from scipy.signal import find_peaks_cwt
from PIL import Image
import os
from glob import glob
import cv2
from keras.models import Model, load_model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from skimage import morphology


class Gui:
    def __init__(self, master):
        self.master = master
        master.title("Python 3 GUI")

        self.label1 = Label(master, text = "x position")
        self.label1.pack()

        self.entry1 = Entry(master)
        self.entry1.pack()

        #entry1.focus_set()

        self.label2 = Label(master, text = "y position")
        self.label2.pack()

        self.entry2 = Entry(master)
        self.entry2.pack()

        self.filepath = tkFileDialog.askopenfilename()
        self.label3 = Label(master, text = "file path")
        self.label3.pack()
        self.label4 = Label(master, text = self.filepath)
        self.label4.pack()
        


        self.button1 = Button(master, text="Run", width=10, command=self.callback)
        self.button1.pack()

    def callback(self):
        print ("the x position is: ", self.entry1.get())
        print ("the y position is: ", self.entry2.get())
        print ("the file name is: ", self.filepath)
        runSeg(self.filepath, self.entry1.get(), self.entry2.get())



def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return ( intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def getbox(image):
    one = np.where(image > 150)
    #print(one)
    if (len(one[0]) != 0):
        vertdiameter = np.max(one[0]) - np.min(one[0])
        horidiameter = np.max(one[1]) - np.min(one[1])
    else:
        vertdiameter, horidiameter = 0, 0
    #print(vertdiameter, horidiameter)
    return (vertdiameter, horidiameter)



def calculateArea(image, horirange, vertirange):
    countarea = []
    countdiameter = []

    xfactor = horirange / float(128) * 2.2
    yfactor = vertirange / float(128) * 1.12

    for i in range(len(image)):
        c = image[i] * 255
        c=c.astype('uint8')
        im = np.squeeze(c)


        #remove small objects using morphology
        im = morphology.remove_small_objects(im, 20)

        area = ((200 < im)&(im < 256)).sum()
        area = area * xfactor * yfactor
        countarea.append(area)

        vertdiameter, horidiameter = getbox(im)
        vertdiameter = vertdiameter * yfactor
        horidiameter = horidiameter * xfactor
        countdiameter.append(vertdiameter)

    countarea = np.asarray(countarea)
    countdiameter = np.asarray(countdiameter)

    return(countarea, countdiameter)

def plotResult(filename, countarea, countdiameter):
    plt.figure(num = None, figsize = (15, 6), dpi = 200)
    plt.plot(countdiameter)
    plt.savefig(filename + "diameter.png")
    plt.gcf().clear()

    plt.figure(num = None, figsize = (15, 6), dpi = 200)
    plt.plot(countarea)
    plt.savefig(filename + "area.png")
    plt.gcf().clear()

def findpeaks(data, step):
    #Find peaks
    peaks = find_peaks_cwt(data, np.arange(1, step))
    peaks2 = find_peaks_cwt(-data, np.arange(1, step))
    #peaks = np.sort(np.asarray(peaks+peaks2))
    plt.figure()
    plt.plot(data, color = 'r')
    high = (np.vstack((peaks, np.asarray(data[peaks]))))
    low = np.vstack((peaks2, np.asarray(data[peaks2])))
    #height = np.flip(np.rot90(height), axis = 0)
    plt.scatter(high[0], high[1])
    plt.scatter(low[0], low[1])
    plt.show()
    plt.gcf().clear()
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


def runSeg(filepath, x, y):
    print(filepath, x, y)
    x = int(x)
    y = int(y)
    START = 0
    END = -1
    HORIRANGE = 80
    VERTIRANGE = 100

    #name = "SHR_S01_HCM1_LA_OD_U-3D_ 4x 0_R03-seg"
    #directory = "Y:/Administrator/Zlab-NAS3/Kate/262018/HCM1+/S01/GPU_processed/" + name + ".tiff"
    directory = filepath


    #Load data
    im = io.imread(directory)
    im = np.asarray(im[START:END, y:y + VERTIRANGE, x:x + HORIRANGE])
    print(im.shape)
    loadmodeldir = '/media/zlab-1/Data/Lian/keras/nData/weights1.h5'

    #Resize data
    X_test = []
    y_test = []
    #for i in range(len(im)):
    for i in range(len(im)):
        X_test.append(cv2.resize(im[i], (128, 128), cv2.INTER_LINEAR))
        #y_test.append(cv2.resize(gt[i], (128, 128), cv2.INTER_LINEAR))
    X_test = np.asarray(X_test)
    X_test=X_test[...,np.newaxis]
    print(X_test.shape)

    #Do segmentation
    K.set_image_data_format('channels_last')
    smooth=1.
    model = load_model(loadmodeldir, custom_objects={'dice_coef_loss':dice_coef,'dice_coef':dice_coef})
    a=model.predict(X_test, batch_size=32, verbose=2)
    print('Total Frame Number:',len(a))

    #Post process
    countarea, countdiameter = calculateArea(im, HORIRANGE, VERTIRANGE)
    plotResult(filepath, countarea, countdiameter)

    heartrate, high_ave, high_std, low_ave, low_std = findpeaks(countdiameter, 15)
    #heartrate, high_ave, high_std, low_ave, low_std = findpeaks(countarea, 15)
    print("Heartrate, EDD, ESD is \n", heartrate, high_ave, low_ave)

    f = open("FlyHeartDat.txt", "a+")
    s = filepath + ", " + str(x) + "," + str(y) + "," + str(heartrate) + ", " + str(high_ave) + ", " + str(low_ave) + "\n"
    f.write(s)
    f.close()


if __name__ == "__main__":
    master = Tk()
    gui = Gui(master)
    master.mainloop()
