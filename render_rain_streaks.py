import numpy as np 
import os
import glob
import cv2
import random
from matplotlib import pyplot as plt
from skimage.filters import gaussian
from skimage.util import img_as_float,img_as_ubyte
from skimage.io import imsave,imread
from skimage.color import rgb2gray


#Otsu's Binarization
def imbinarize_Simple(img, threshold_min=50,threshold_max=225):
    _, th = cv2.threshold(img, threshold_min, threshold_max, cv2.THRESH_BINARY)
    return th

#Otsu's Binarization
def imbinarize_O(img, GaussianBlur=False):
    if GaussianBlur:
        img = cv2.GaussianBlur(img,(3,3),0)
    _,th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th

def imbinarize_O_float(img, GaussianBlur=False):
    if GaussianBlur:
        img = cv2.GaussianBlur(img,(3,3),0)
    _,th = cv2.threshold(img,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th

#Adaptive Mean Thresholding
def imbinarize_M(img):
    th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    return th
#Adaptive Gaussian Thresholding
def imbinarize_G(img, GaussianBlur=False):
    th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    return th

def bwareafilt(image,min_size=0,max_size=1000,connectivity=4):
    image = image.astype(np.uint8)
    nb_components, labels , stats, _ = cv2.connectedComponentsWithStats(image, connectivity=connectivity)
    sizes = stats[:, -1]
    img2 = np.zeros(labels.shape)
    label = 1
    for i in range(2,nb_components):
        if sizes[i] <= max_size and sizes[i]>=min_size:
            label = i
            img2[labels == label] = 1

    return img2

def show_imgs_in_cols(rows,cols,img_list,title_list=[],figsize=(15, 10)):
    total = rows*cols
    fig = plt.figure(figsize=figsize)
    for i in range(min(len(img_list),total)):
        plt.subplot(rows,cols,i+1)
        plt.imshow(img_list[i])
        if len(title_list):
            plt.title(title_list[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()
        
def write2txt(fname,strings,mode='a'):
    with open(fname,mode) as f:
        f.write(strings)

def add_rain_streak(im,str_index=4,stype='dense',gaussian_sigma = 2,st_min_size = 0,st_max_size = 1000 ,streak_root = 'Streaks_Garg06/'):  
    if stype=='dense':
        steps = 8
        iter_num = 0
        sigma = 0.2
        mu = 0.25            
    elif stype == 'middle':
        steps = 4
        iter_num = 6
        sigma = 0.2
        mu = 0.25
           
    elif stype == 'sparse':
        steps = 2
        iter_num = 6
        sigma = 0.1
        mu = 0.1

    else:
        print("streak types only valid for 'dense','middle','sparse' ")
        return
    
    #filename = im_path.split(os.path.sep)[-1].split('.')[0]
    bh = im.shape[0]
    bw = im.shape[1]
    #rain_final = img_as_float(im)
    stage_rain_final = img_as_float(im)
    #st_final = np.zeros((bh,bw,3)).astype(float)
    str_file_list = glob.glob(streak_root+'*-{}.png'.format(str_index))
    stage_st_final = np.zeros((bh,bw,3)).astype(float)
    for i in range(steps):
        strnum = random.randint(0,len(str_file_list)-1)
        st = cv2.imread(str_file_list[strnum])
        resize_shape = (int(st.shape[1]*4),int(st.shape[0]*4))
        st = cv2.resize(st,resize_shape)
        newst = np.zeros(st.shape).astype(float)
        img = cv2.cvtColor(st,cv2.COLOR_BGR2GRAY)
        bwst = imbinarize_O(img,True)
        mask = bwareafilt(bwst,st_min_size,st_max_size)
        for c in range(3):
            temp = st[:,:,c]
            temp = np.multiply(temp,mask.astype(np.uint8))
            newst[:,:,c] = gaussian(temp,gaussian_sigma,truncate=2)
        if not iter_num:     
            newst = cv2.resize(newst, (bw, bh))
            tr = random.random() * sigma + mu
            #rain_final = rain_final + newst.astype(float) * tr
            stage_rain_final = stage_rain_final + newst.astype(float) * tr
            #st_final = st_final + newst.astype(float)*tr
            stage_st_final = stage_st_final + newst.astype(float)*tr
        else:
            newst = cv2.resize(newst, (int(newst.shape[1]/2),int(newst.shape[0]/2)))
            sh = newst.shape[0] 
            sw = newst.shape[1]
            
            for iter in range(iter_num):
                row = random.randint(1,sh - bh)
                col = random.randint(1,sw - bw)
                selected = newst[row:row+bh, col:col+bw, :]
                tr = random.random() * sigma + mu
                #rain_final = rain_final + selected.astype(float) * tr
                stage_rain_final = stage_rain_final + selected.astype(float) * tr
                #st_final = st_final + selected.astype(float)*tr
                stage_st_final = stage_st_final + selected.astype(float)*tr
    
    stage_st_final = np.clip(stage_st_final,0.0,1.0)
    #stage_st_final = rgb2gray(stage_st_final)
    pic_name = 'out/{0}-str-type{1}.png'.format(stype,str_index)
    cv2.imwrite(pic_name,img_as_ubyte(stage_st_final))
    #write2txt(stype_list_stats,pic_name,'a')
    stage_rain_final = np.clip(stage_rain_final,0.0,1.0)
    pic_name = 'out/{0}-rain-type{1}.png'.format(stype,str_index)
    cv2.imwrite(pic_name,img_as_ubyte(stage_rain_final))
    pic_name = 'out/{0}-norain-type{1}.png'.format(stype,str_index)
    cv2.imwrite(pic_name,im)
    #write2txt(rain_stype_list_stats,pic_name,'a')
    return stage_rain_final,stage_st_final


    
if __name__ == "__main__":
    
    image_root = 'no-rain/'
    img_file_list = glob.glob(image_root+'*.png')
    num_of_strtype = 5
    num_of_file = len(img_file_list)

    fileindex = 0
    im_path = img_file_list[fileindex]
    im = cv2.imread(im_path)
    stypes = ['dense','middle','sparse']
    for str_index in range(4,9):
        for stype in stypes:
            rain,streaks=add_rain_streak(im,str_index,stype)

    