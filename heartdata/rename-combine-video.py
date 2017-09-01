# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 09:58:52 2017

@author: qinxi
"""
from PIL import Image
import numpy as np
import os
from glob import glob
import cv2

folder='F:/heartdata/result4/'
files=glob(folder+'*.jpg')
for j in files:
    name=os.path.basename(j)
    if name.replace('.jpg','').isdigit() == True:
        os.rename(os.path.join(folder,name),os.path.join(folder, name.replace('.jpg','').zfill(4)+'.jpg'))
    if name.replace('.jpg','').isdigit() == False:
        os.rename(os.path.join(folder,name), os.path.join(folder,name.replace('_original.jpg','').zfill(4)+'_original.jpg'))
    
    
    
folder='F:/heartdata/result9/'
files=glob(folder+'*.jpg')
for j in range(len(files)//2):
    img=Image.open(files[j*2])
    img1=Image.open(files[j*2+1])
    new_im = Image.new('RGB', (256,128))
    new_im.paste(img1, (0,0))
    new_im.paste(img, (128,0))
    new_im.save("F:/heartdata/result9/combine/{}.png".format(format(j,'05')))
    print(j)


files1=glob("F:/heartdata/result2/combine/*.png")
files2=glob("F:/heartdata/result3/combine/*.png")
for j in range(len(files1)):
    img1=Image.open(files1[j])
    img2=Image.open(files2[j])
    new_im = Image.new('RGB', (256,256))
    new_im.paste(img1, (0,0))
    new_im.paste(img2, (0,128))
    new_im.save("F:/heartdata/result23/{}.png".format(format(j,'05')))
    print(j)





from moviepy import editor 
gif_name = 'pic'
fps = 50
folder='F:/heartdata/result9/combine/'
file_list = glob(folder+'*.png') 
clips = [editor.ImageClip(m).set_duration(0.1) for m in file_list]
concat_clip = editor.concatenate_videoclips(clips, method="compose")
concat_clip.write_videofile("result9.mp4", fps=fps)