# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:38:41 2017

@author: Zlab-6
"""

from PIL import Image
import numpy as np
import os
import cv2
from skimage import transform
from matplotlib import pyplot as plt
#from peakdetect import peakdetect
from scipy.signal import find_peaks_cwt
from flynetfunctions import findpeaks
import pandas as pd

def plotbar(highgt, lowgt, highstdgt, lowstdgt, ylabel):
    n_groups = 3
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    error_config = {'ecolor': '0.3'}
    rects1 = plt.bar(index + bar_width, highgt, bar_width, color = '#3f51b5', yerr = highstdgt, error_kw=error_config, label = 'Manully Labeling')
    rects2 = plt.bar(index + 2 * bar_width, lowgt, bar_width, color = '#ff7043', yerr = lowstdgt, error_kw=error_config, label = 'Model Prediction')
    plt.xlabel('Types of fly')
    plt.ylabel(ylabel)
    plt.xticks(index + 1.5 * bar_width, ('Larva', 'Pupa', 'Adult'))
    plt.legend()
    plt.tight_layout()
    plt.savefig('./resultimage/fig4_'+ylabel+'.png')
    #plt.show()



def processpeak(i):

    #Load data
    data_ad = np.load('./resultimage/diameterAD_'+ str(i)+'.npy')
    data_ep = np.load('./resultimage/diameterEP_'+ str(i)+'.npy')
    data_la = np.load('./resultimage/diameterLarva_'+ str(i)+'.npy')
    high_ave = []
    low_ave = []
    high_std = []
    low_std = []
    highgt = []
    lowgt = []
    highstdgt = []
    lowstdgt = []
    highpd = []
    lowpd = []
    highstdpd = []
    lowstdpd = []
    heartrategt = []
    heartratepd = []
    for data1 in [data_la, data_ep, data_ad]:
        #data1[2] is useless
        for i in range(2):
            data = data1[i]
            step = 40
            if (np.array_equal(data1, data_ad)):
                step = 15
            heartrate1, high_ave1, high_std1, low_ave1, low_std1 = findpeaks(data, step)
            if i ==1:
                highgt.append(high_ave1)
                lowgt.append(low_ave1)
                highstdgt.append(high_std1)
                lowstdgt.append(low_std1)
                heartrategt.append(heartrate1)
            else:
                highpd.append(high_ave1)
                lowpd.append(low_ave1)
                highstdpd.append(high_std1)
                lowstdpd.append(low_std1)
                heartratepd.append(heartrate1)
    ratiogt=[(highgt[_]-lowgt[_])/highgt[_] for _ in range(3)]
    ratiopd=[(highpd[_]-lowpd[_])/highpd[_] for _ in range(3)]
    '''
    plotbar(highgt, highpd, highstdgt, lowstdgt, "End Diastolic Diameter")
    plotbar(lowgt, lowpd, highstdgt, lowstdgt, "End Systolic Diameter")
    plotbar(heartrategt, heartratepd, highstdgt, lowstdgt, "Heart Rate")
    plotbar(ratiogt, ratiopd , [0,0,0], [0,0,0], "Fraction Shortening")
    '''
    return (highgt, highpd, highstdgt, lowstdgt, lowgt, lowpd, heartrategt, heartratepd, ratiogt, ratiopd)



    #d1=np.vstack((np.asarray([highgt,highpd]),np.asarray([lowgt,lowpd]),np.asarray([heartrategt,heartratepd]),np.asarray([ratiogt,ratiopd])))

    #dd1=pd.DataFrame(d1,columns=['larva','pupa','adult'], index=['highgt','highpd','lowgt','lowpd','rategt','ratepd','ratiogt','ratiopd'])

    #save
    #dd1.to_csv('./peak_data.csv',index=True)

    #load
    #a=pd2['larva']['highgt']

    #print(a.mean())


if __name__ == '__main__':

    for i in range(5,14):
        highgt, highpd, highstdgt, lowstdgt, lowgt, lowpd, heartrategt, heartratepd, ratiogt, ratiopd=processpeak(i)
        df1=pd.read_csv('./peak_data.csv',index_col=0)

        #append
        d2=np.vstack((np.asarray([highgt,highpd]),np.asarray([lowgt,lowpd]),np.asarray([heartrategt,heartratepd]),np.asarray([ratiogt,ratiopd])))
        dd2=pd.DataFrame(d2,columns=['larva','pupa','adult'], index=['highgt','highpd','lowgt','lowpd','rategt','ratepd','ratiogt','ratiopd'])

        pd2=pd.concat([df1,dd2])
        pd2.to_csv('./peak_data.csv',index=True)














'''
d=[[highgt,highpd],[lowgt,lowpd],[heartrategt,heartratepd],[ratiogt,ratiopd]]
df=pd.DataFrame(data=d,columns=['groundtruth','prediction'],index=['high','low','rate','ratio'])
df.append(df).to_csv('df.csv')

df1=pd.read_csv('df.csv')
print(df1.shape)
print(df1.iloc[0].shape)
print(df1.iloc[4].shape)
'''
'''
n_groups = 3
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
error_config = {'ecolor': '0.3'}
rects1 = plt.bar(index + bar_width, highgt, bar_width, color = '#3f51b5', yerr = highstdgt, error_kw=error_config, label = 'GroundTruth')
rects2 = plt.bar(index + 2 * bar_width, lowgt, bar_width, color = '#ff7043', yerr = lowstdgt, error_kw=error_config, label = 'Prediction')
plt.xlabel('Types of fly')
plt.ylabel('Heart Diameter')
plt.xticks(index + 2 * bar_width, ('larva', 'EP', 'AD'))
plt.legend()
plt.tight_layout()
plt.show()
'''
