#LD

import numpy as np
from skimage import io
import os
from glob import glob

def getBox(image, xrange1, yrange):
    one = np.where(image > 150)
    centerx = (one[1].max() + one[1].min())/2
    centery = (one[0].max() + one[0].min())/2
    return (centerx-xrange1, centerx + xrange1, centery - yrange, centery + yrange)

def cutImage(imagefile, maskfile):
    print(imagefile)
    print(maskfile)
    image = io.imread(imagefile)
    mask = io.imread(maskfile)
       
    
    xrange1 = 64
    yrange = 100
    os.makedirs("/media/zlab-1/Data/Lian/keras/nTrain/newAD/{}".format(os.path.basename(imagefile)))

    for i in range(len(mask)):
        if mask[i,:,:,0].sum() > 0:
            xstart, xend, ystart, yend = getBox(mask[i,:,:,0], xrange1, yrange)
            io.imsave("/media/zlab-1/Data/Lian/keras/nTrain/newAD/{}/{}_mask.png".format(os.path.basename(imagefile), format(i,'05')), np.uint8(mask[i, ystart:yend, :, 0]))
            io.imsave("/media/zlab-1/Data/Lian/keras/nTrain/newAD/{}/{}.png".format(os.path.basename(imagefile), format(i,'05')), image[i, ystart:yend, :])
    

    #im1.save("/media/zlab-1/Data/QX/extract/{}/{}_mask.png".format(os.path.basename(file[0]),l))
    

    #image = Image.open(imagefile)
    #mask = Image.open(maskfile)

if __name__ == '__main__':
    path = "/run/user/1000/gvfs/smb-share:server=128.180.65.173,share=data/Lian/flyheart/newdata/masked/"
    imagefile = path + "SHR_S12_HCM1+_LA_OD_U-3D_ 4x 0_R01.tiff"
    maskfile = path + "SHR_S12_HCM1+_LA_OD_U-3D_ 4x 0_R01.Labels.tif"
    example = "/media/zlab-1/Data/Lian/keras/nTrain/Larva/2014-09-12_lava_S01/00598_mask.png"

    '''
    sample = mask[0,:,:,0]
    sample = np.uint8(sample)
    io.imsave("./sample.png",sample)
    readsample = io.imread("./sample.png")

    exampleimage = io.imread(example)
    '''



    imagefiles = sorted(glob(path + "*.tiff"))
    maskfiles = sorted(glob(path + "*.tif"))

    for i in range(len(imagefiles)):
        cutImage(imagefiles[i], maskfiles[i])
    #cutImage(imagefile, maskfile)    
    



