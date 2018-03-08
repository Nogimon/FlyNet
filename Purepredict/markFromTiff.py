import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from scipy.signal import find_peaks_cwt

def getbox(image):
    one = np.where(image > 0.5)
    if (len(one[0]) != 0):
        vertdiameter = np.max(one[0]) - np.min(one[0])
        horidiameter = np.max(one[1]) - np.min(one[1])
    else:
        vertdiameter = 0
        horidiameter = 0
    i = 0
    
    return (vertdiameter, horidiameter)

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


def plotFigure(diametervd, name):
    plt.figure(num = None, figsize = (15, 6), dpi = 200)
    plt.plot(diametervd)
    plt.ylabel("diameter")
    plt.xlabel("frames")
    #plt.show()

    plt.savefig('./markFromTiff/' + name + '.png')
    plt.gcf().clear()




if __name__ == "__main__":
    
    #Parameters to set
    name = "fly-1-correct-seg"
    directory = "./" + name + ".tiff"
    START =1800
    END = 3000
    YFACTOR = 1.12
    XFACTOR = 2.2
    TIMEFACTOR = 129.9


    im = io.imread(directory, as_grey=True)

    im = np.asarray(im[START:END, :, :])


    diametervd = []
    diameterhd = []
    for i in range(0, len(im)):
        vertdiameter, horidiameter = getbox(im[i])
        diametervd.append(vertdiameter)
        diameterhd.append(horidiameter)

    diametervd = np.asarray(diametervd)
    diameterhd = np.asarray(diameterhd)

    diametervd = diametervd * YFACTOR
    diameterhd = diameterhd * XFACTOR


    plotFigure(diametervd, name)
    
    heartrate, high_ave, high_std, low_ave, low_std = findpeaks(diametervd, 30)

    print("Heartrate, EDD, ESD is \n", heartrate, high_ave, low_ave)
