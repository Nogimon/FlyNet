import numpy as np
from skimage import io
from skimage import morphology
from matplotlib import pyplot as plt
from scipy.signal import find_peaks_cwt
from glob import glob
from PIL import Image

'''
class Parameters(x0, y0):
    #y direction um/pixel
    yfactor = 1.12 * 200 / 128
    #x direction um/pixel
    xfactor = 2.2 * 80 / 128
    #time frames/second
    timefactor = 129.9
'''

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
        im = image[i]

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

def plotResult(name, countarea, countdiameter, directory):
    plt.figure(num = None, figsize = (15, 6), dpi = 200)
    plt.plot(countdiameter)
    plt.savefig(directory  + "/diameter.png")
    plt.gcf().clear()

    plt.figure(num = None, figsize = (15, 6), dpi = 200)
    plt.plot(countarea)
    plt.savefig(directory  + "/area.png")
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


def loaddata(files):
    im = []
    for i in files:
        image = np.asarray(Image.open(i))
        im.append(image)
    im = np.asarray(im)
    return im

if __name__ == "__main__":
    START = 0
    END = -1
    HORIRANGE = 128
    VERTIRANGE = 128

    #name = "SHR_S01_HCM1_LA_OD_U-3D_ 4x 0_R03-seg"
    #directory = "Y:/Administrator/Zlab-NAS3/Kate/262018/HCM1+/S01/GPU_processed/" + name + ".tiff"
    #directory = "D:/Lian/images/" + name + ".tiff"
    #im = io.imread(directory)
    #im = im[START:END]

    im = []
    directory = "/run/user/1000/gvfs/smb-share:server=128.180.65.184,share=home/Zlab-NAS3/Kate/262018/Larva/HCM2+/S02/"
    name = directory[-10:]
    files = sorted(glob(directory + "*/*[0-9].jpg"))
    im = loaddata(files)





    countarea, countdiameter = calculateArea(im, HORIRANGE, VERTIRANGE)
    plotResult(name, countarea, countdiameter, directory)

    heartrate, high_ave, high_std, low_ave, low_std = findpeaks(countdiameter, 15)
    #heartrate, high_ave, high_std, low_ave, low_std = findpeaks(countarea, 15)
    print("Heartrate, EDD, ESD is \n", heartrate, high_ave, low_ave)

    f = open("FlyHeartDat.txt", "a+")
    s = name + ", " + str(heartrate) + ", " + str(high_ave) + ", " + str(low_ave) + "\n"
    f.write(s)
    f.close()
