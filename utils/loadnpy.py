import numpy as np
from PIL import Image
from skimage import morphology

a = np.load('./bboxtry/595.npy')
a = morphology.remove_small_objects(a, 350)
im = Image.fromarray(a)
b=im.getbbox()
aa=a[b[1]:b[3],b[0]:b[2]]
img=Image.fromarray(aa)
img.show()
im.show()