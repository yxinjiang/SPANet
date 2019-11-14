'''
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.
Written by lizhi@h2i.sg, Date June 17, 2019
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import dateutil
from glob import glob
from utils import addrain, add_rain,autocrop_night, autocrop_day, normalize
import scipy.io
import os
import h5py
import torch.utils.data as udata
import random
import torch
import random
from render_rain_streaks import add_rain_streak
from skimage.util import img_as_uint


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)

    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1

    return Y.reshape([endc, win, win, TotalPatNum])

def read_video(video_name):
	# where videos are stored:
   # change if video stores somewhere else
	video= cv2.VideoCapture(video_name)
	return video

def video2image(num_frames, video_name,rows=slice(0,400), cols=slice(0,300),CROP=True):
    video= read_video(video_name)
    Frames= []
    ind=0
    while True:
        video.set(cv2.CAP_PROP_POS_MSEC,ind*1000)
        ret, frame= video.read()
		#cv2.imwrite('video_frames/'+video_name+'_frame0.jpg',frame)
            
        if ind==num_frames or not ret:
            break
        if CROP:
            Frames.append(frame[rows,cols])
        else:
            Frames.append(frame)

            #Frames.append(frame[int(frame.shape[0]/2):,:int(frame.shape[1]/2)])
        ind+=1

    Frames= np.array(Frames)
    print(Frames.shape)
    return Frames.transpose(1,2,3,0)

def rand_video2image(num_frames, video_name, store_img=False):
	video= read_video(video_name)
	Frames= []
	ind=-1
	n_rand= np.random.choice(range(310),100, replace=False)
	while True:
		ind+=1
		ret, frame= video.read()
		if len(Frames)==num_frames or not ret:
			break
		elif ind in n_rand:
			Frames.append(frame)
	print('%d frames of video extracted...'%num_frames)
	Frames= np.array(Frames)

	return Frames

def pick_video(path):
	video_types = ['*.ts','*.mkv','*.mp4']
	videos = []
	for v in video_types:
		videos.extend(glob(os.path.join(path, v)))
	video= np.random.choice(videos)
	print(video)

	return video

def syn_test(video_path,store_syn_img=False,rows=slice(0,400), cols=slice(0,300)):
	save_path = os.path.join(video_path,'datasets')
	video= pick_video(video_path)
	Frames= rand_video2image(1000, video,store_img=True)
	assert len(Frames.shape)==4,f"The shape of Frames is not consistent, expected 4 but {len(Frames.shape)} received"
	new_shape= (Frames.shape[0],400,300,3)
	# print(new_shape)
	# print('auto cropping image')
	# croped_rows, croped_cols= autocrop_day(cv2.cvtColor(Frames[0,:,:,:], cv2.COLOR_BGR2GRAY), window_size)
	# print(f'croped image: {croped_rows, croped_cols}')

	new_frames= np.zeros(new_shape, dtype=np.float32)
	new_frames= Frames[:, 600:1000, 0:300,:].copy()
	synthetic_img = np.zeros(new_shape, dtype=np.float32)
	rain_layer = np.zeros(new_shape, dtype=np.float32)
	for i in range(Frames.shape[0]):
		frame=  new_frames[i,:,:,:].copy()
		
		#plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
		#plt.show()
		'''
		deg = random.randint(45,91)
		alpha = random.randint(6,9)*0.1
		synthetic_img, rain_layer= addrain(frame,deg,alpha)
		'''

		'''
		rain_slant="Numeric value between -45 and 45 is allowed"
		rain_width="Width value between 1 and 5 is allowed"
		rain_length="Length value between 0 and 100 is allowed"
		drop_color = "Drop color between 150 and 254 is allowed"
		rain_type = ['drizzle','heavy','torrential']
		'''
		'''
		rain_slant = random.randint(-45,46)
		rain_width = random.randint(1,6)
		rain_l_w_ratio = random.randint(10,31)
		rain_length = rain_width*rain_l_w_ratio
		drop_color = random.randint(150,255)
		rain_type = random.sample(['drizzle','heavy','torrential'],1)[0]
		#(image,slant=-1,drop_length=20,drop_width=1,drop_color=(200,200,200),rain_type='None')
		synthetic_img[i,:,:,:], single_rain_layer= add_rain(frame,rain_slant,rain_length,rain_width,(drop_color,drop_color,drop_color),rain_type)
		single_rain_layer = cv2.cvtColor(single_rain_layer,cv2.COLOR_RGB2GRAY)
		'''
		str_index = random.randint(4,8)
		single_synthetic_img, single_rain_layer = add_rain_streak(frame,str_index,stype='middle')

		#BGR to YCrCb
		#new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
		#single_synthetic_img = cv2.cvtColor(img_as_uint(single_synthetic_img), cv2.COLOR_BGR2YCrCb)
		#single_rain_layer = cv2.cvtColor(img_as_uint(single_rain_layer), cv2.COLOR_BGR2YCrCb)
		new_frames[i,:,:,:] = frame
		synthetic_img[i,:,:,:] = single_synthetic_img
		rain_layer[i,:,:,:] = single_rain_layer
		if store_syn_img:
			img_seq = f'-{i}.png'
			cv2.imwrite(os.path.join(save_path,'rain','rain'+img_seq), single_synthetic_img)
			cv2.imwrite(os.path.join(save_path,'no-rain','norain'+img_seq), frame)
			cv2.imwrite(os.path.join(save_path,'streak','streak'+img_seq), single_rain_layer)

	print('Synthetic rainfall added ...')

	return new_frames, synthetic_img,rain_layer
	#save as the .mat format
	# scipy.io.savemat('C:\\Radar Projects\\lizhi\\CCTV\\Videos\\20190110191017-rain.mat', mdict={'Rain': rain_streaks})
	# scipy.io.savemat('C:\\Radar Projects\\lizhi\\CCTV\\Videos\\20190110191017-img.mat', mdict={'Rain': new_frames})

def pyh5(win,video_path,rows=slice(0,400), cols=slice(0,300)):
	rain_img= np.zeros((300,400,300,3), np.float32)
	norain_img= np.zeros((300,400,300,3), np.float32)
	rain_layer= np.zeros((300,400,300,3), np.float32)
	save_path = os.path.join(video_path,'datasets')
    
	for i in range(3):
		norain_img[i*100:(i+1)*100,:,:,:], rain_img[i*100:(i+1)*100,:,:,:],rain_layer[i*100:(i+1)*100,:,:] = syn_test(video_path, True,rows,cols)
	save_input_path= os.path.join(save_path, 'train_input.h5')
	save_target_path= os.path.join(save_path, 'train_target.h5')
	save_target_streak_path= os.path.join(save_path, 'train_streak_target.h5')

	target_h5f= h5py.File(save_target_path,'w')
	target_streak_h5f= h5py.File(save_target_streak_path,'w')
    
	input_h5f= h5py.File(save_input_path, 'w')
	train_num=0

	for i in range(len(rain_img)):

		norain_img= np.float32(normalize(norain_img))
		rain_img= np.float32(normalize(rain_img))
		rain_layer= np.float32(normalize(rain_layer))

		
		input_patches= Im2Patch(rain_img[i,:,:,:].transpose(2,0,1), win, 80)

		target_patches= Im2Patch(norain_img[i,:,:,:].transpose(2,0,1), win, 80)
        
		target_streak_patches= Im2Patch(rain_layer[i,:,:,:].transpose(2,0,1), win, 80)

		for n in range(target_patches.shape[-1]):
			target_data = target_patches[:, :, :, n].copy()
			target_h5f.create_dataset(str(train_num), data=target_data)

			target_streak_data = target_streak_patches[:, :, :, n].copy()
			target_streak_h5f.create_dataset(str(train_num), data=target_streak_data)

			input_data = input_patches[:, :, :, n].copy()
			input_h5f.create_dataset(str(train_num), data=input_data)
			print("# samples: %d" % (target_patches.shape[3]))
			train_num += 1

	print("total trainning samples ", train_num)
	target_h5f.close()
	input_h5f.close()

class Dataset(udata.Dataset):
	def __init__(self, data_path='.'):
		super(Dataset, self).__init__()

		self.data_path = data_path

		target_path = os.path.join(self.data_path, 'train_target.h5')
		target_streak_path = os.path.join(self.data_path, 'train_streak_target.h5')

		input_path = os.path.join(self.data_path, 'train_input.h5')

		target_h5f = h5py.File(target_path, 'r')
		target_streak_h5f = h5py.File(target_streak_path, 'r')

		input_h5f = h5py.File(input_path, 'r')

		self.keys = list(target_h5f.keys())
		random.shuffle(self.keys)

		target_h5f.close()
		target_streak_h5f.close()

		input_h5f.close()

	def __len__(self):
		return len(self.keys)

	def __getitem__(self, index):

		target_path = os.path.join(self.data_path, 'train_target.h5')
		target_streak_path = os.path.join(self.data_path, 'train_streak_target.h5')
		input_path = os.path.join(self.data_path, 'train_input.h5')

		target_h5f = h5py.File(target_path, 'r')
		target_streak_h5f = h5py.File(target_streak_path, 'r')

		input_h5f = h5py.File(input_path, 'r')

		key = self.keys[index]
		target = np.array(target_h5f[key])
		target_streak = np.array(target_streak_h5f[key])

		input = np.array(input_h5f[key])

		target_h5f.close()
		target_streak_h5f.close()

		input_h5f.close()

		return torch.Tensor(input), torch.Tensor(target), torch.Tensor(target_streak)

if __name__ =='__main__':
	#save_path = './video'
	video_path = 'D:\\CCTV_Task2\\Arlo\\Ultra4k_TestVideos\\NoRain'
	rows=slice(600,1000)
	cols=slice(600,900)
	save_path = os.path.join(video_path,'datasets')
	rain_path = os.path.join(save_path,'rain')
	norain_path = os.path.join(save_path,'no-rain')
	streak_path = os.path.join(save_path,'streak')
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	if not os.path.exists(rain_path):
		os.mkdir(rain_path)
	if not os.path.exists(norain_path):
		os.mkdir(norain_path)
	if not os.path.exists(streak_path):
		os.mkdir(streak_path)
	pyh5(100,video_path,rows,cols)