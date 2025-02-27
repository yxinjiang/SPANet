'''
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.
Written by lizhi@h2i.sg, Date June 17, 2019
'''

import math
import torch
import re
import torch.nn as nn
import numpy as np
from skimage.measure.simple_metrics import compare_psnr
import  os
import glob
import cv2
from numba import jit
import Automold as am 
import Helpers as hp  

def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*epoch(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])


def normalize(data):
    return data / 255.


def is_image(img_name):
    if img_name.endswith(".jpg") or img_name.endswith(".bmp") or img_name.endswith(".png"):
        return True
    else:
        return  False


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def autocrop_day(src, window_size=(300,300)):
    #with numba, the cost reduces from 92 seconds to 29 seconds.
    @jit(nopython=True)
    def moving_window(h,w, min_val, window_size):
    # val= overdetection(src, 2)
    # src[src<=val]=0
    # src[src>val]=255
        for i in range(h-window_size[0]):
            for j in range(w-window_size[1]):
                tot= src[i:window_size[0]+i, j:j+window_size[1]].sum()
                if tot<min_val:
                    min_val= tot
                    rows= (i, window_size[0]+i)
                    cols= (j,j+window_size[1])

        return rows, cols
    src= cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    h,w= src.shape
    min_val= np.inf

    rows, cols= moving_window(h,w,min_val,window_size)

    return rows, cols

def autocrop_night(src, window_size=(300,300)):
    h,w = src.shape
    max_val= 0
    # val= overdetection(src, 2)
    # src[src<=val]=0
    # src[src>val]=255
    for i in range(h-window_size[0]):
        for j in range(w-window_size[1]):
            tot= src[i:window_size[0]+i, j:j+window_size[1]].sum()
            if tot>max_val: max_val=tot; rows=slice(i, window_size[0]+i); cols=slice(j,j+window_size[1])

    return rows, cols

def addrain(src, deg=90, alpha=0.6):  #deg: 45~90
    #leave it with vertical rainfall streaks
    # add gaussian noise
    noise= np.random.normal(loc=0, scale=0.5*255, size=src.shape[:2]).astype(np.uint8)
    # initialize motion blur kernel
    size=25 #25
    beta = 1 - alpha
    motion_blur_kernel= np.zeros((size, size))
    motion_blur_kernel[:, int((size-1)/2)]= np.ones(size)
    motion_blur_kernel/=size
    rain_layer= cv2.filter2D(noise, -1, motion_blur_kernel).astype(np.uint8)
    if len(src.shape)==3:
        for channel in range(3):
            src[:,:,channel]= alpha*src[:,:,channel]+ beta*rain_layer
    else:
        src= alpha*src+ beta*rain_layer
    rain_layer = np.expand_dims(rain_layer,3)
    return src.astype(np.uint8), rain_layer.astype(np.uint8)

##rain_type = ['drizzle','heavy','torrential']
##slant = [-20,20]
## drop_length = [0,100]
## drop_width = [1, 5]
def add_rain(src,slant=-1,drop_length=20,drop_width=1,drop_color=(200,200,200),rain_type='None'): ## (200,200,200) a shade of gray
    rain_image,rain_streak = am.add_rain(src,slant=-1,drop_length=20,drop_width=1,drop_color=(200,200,200),rain_type='None')
    return rain_image,rain_streak


def overdetection(img, window_size):
    h,w= img.shape
    length= int(h/window_size)
    tot=0
    for i in range(window_size):
        for j in range(window_size):
            tot+= img[i*length:(i+1)*length, j*length:(j+1)*length].mean()
    return tot/window_size/window_size