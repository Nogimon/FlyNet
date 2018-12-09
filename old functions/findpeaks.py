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
    return (high_ave, high_std, low_ave, low_std)

#Load data
data_ad = np.load('./resultimage/Adult.npy')
data_ep = np.load('./resultimage/EP.npy')
data_la = np.load('./resultimage/larva.npy')
high_ave = []
low_ave = []
high_std = []
low_std = []
for data in [data_la, data_ep, data_ad]:
    step = 40
    if (np.array_equal(data, data_ad)):
        step = 15
    high_ave1, high_std1, low_ave1, low_std1 = findpeaks(data, step)
    high_ave.append(high_ave1)
    high_std.append(high_std1)
    low_ave.append(low_ave1)
    low_std.append(low_std1)


n_groups = 3
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
error_config = {'ecolor': '0.3'}
rects1 = plt.bar(index + bar_width, high_ave, bar_width, color = '#3f51b5', yerr = high_std, error_kw=error_config, label = 'End Diastolic Area')
rects2 = plt.bar(index + 2 * bar_width, low_ave, bar_width, color = '#ff7043', yerr = low_std, error_kw=error_config, label = 'End Systolic Area')
plt.xlabel('Types of fly')
plt.ylabel('Heart Diameter')
plt.xticks(index + 2 * bar_width, ('larva', 'EP', 'AD'))
plt.legend()
plt.tight_layout()
plt.show()
#plt.savefig('heartdiameter.png')

