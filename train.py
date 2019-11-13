'''
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.
Written by lizhi@h2i.sg, Date June 17, 2019
'''

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import sys
#sys.path.append('./PReNet')
from dataprep import *
from utils import *
from torch.optim.lr_scheduler import MultiStepLR
from SSIM import SSIM
from generator import *
#from networks import *
import timeit
from skimage.util import img_as_uint,img_as_float
import cv2
from render_rain_streaks import imbinarize_O



if torch.cuda.is_available():
	print('using gpu training ...')

def main():
	print('Loading dataset ...\n')
	dataset_train= Dataset(data_path= 'datasets_arlo/')
	loader_train= DataLoader(dataset= dataset_train)
	print('# of training samples :', int(len(loader_train)))
	# define some hyper-parameters
	recurr_iter= 4
	use_GPU= True
	model_path= 'logs/real/latest.pth'
	num_epochs= 5

	#model = PRN(recurr_iter, use_GPU)


	#model= Generator_lstm(recurr_iter, use_GPU)
	model= Generator_lstm_mask(recurr_iter, use_GPU)
	print_network(model)
	model.load_state_dict(torch.load(model_path))

	#loss
	L1 = nn.L1Loss(size_average=False)
	L2 = nn.MSELoss(size_average=False)
	binary_cross_entropy = F.binary_cross_entropy
	
	criterion= SSIM()

	if use_GPU:
		model= model.cuda()
		L1.cuda()
		L2.cuda()
		criterion.cuda()
		#binary_cross_entropy.cuda()

	#optimizer:
	optimizer= optim.Adam(model.parameters(), lr= 1e-4)
	scheduler = MultiStepLR(optimizer, milestones=[20,40], gamma=0.1)

	#record training
	writer= SummaryWriter('logs/')
	step=0
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		for i, (input_train, target_train, streak_train) in enumerate(loader_train,0):
			input_train, target_train, streak_train = Variable(input_train, requires_grad=False), Variable(target_train, requires_grad=False), Variable(streak_train, requires_grad=False)
			if use_GPU:
				input_train, target_train, streak_train= input_train.cuda(), target_train.cuda(),streak_train.cuda()
			optimizer.zero_grad()
			model.train()
			out_streak,mask,_ =model(input_train)
			out_train = input_train - out_streak
			out_streak= torch.clamp(out_streak[:,:,:,:], 0., 1.)
			
			pixel_metric= L2(out_streak[:,0,:,:],streak_train[:,0,:,:]) + L2(mask[:,0,:,:], streak_train[:,0,:,:])#+ (1- criterion(target_train,out_train)) #L1(streak_train[:,0,:,:], out_streak[:,0,:,:]) + 
			#loss= -pixel_metric
			loss = pixel_metric

			loss.backward()
			optimizer.step()

			model.eval()
			out_streak,mask,_ = model(input_train)
			out_streak_gray = torch.clamp(out_streak[:,0,:,:], 0., 1.)
			print("[epoch %d][%d/%d] loss: %.4f, pixel_metric: %.4f" %
                  (epoch+1, i+1, len(loader_train), loss.item(), pixel_metric.item()))

			if step%10==0:
				writer.add_scalar('loss', loss.item(), step)
			step+=1

		model.eval()
		out_streak,mask,_ = model(input_train)
		out_streak_gray= torch.clamp(out_streak[:,0,:,:],0.,1.)
		im_target= utils.make_grid(target_train.data, nrow=8, normalize=True, scale_each=True)
		im_input= utils.make_grid(input_train.data, nrow=8, normalize=True, scale_each=True)
		out_streak_gray= utils.make_grid(out_streak_gray.data, nrow=8, normalize=True, scale_each=True)
		writer.add_image('clean image', im_target, epoch+1)
		writer.add_image('rainy image', im_input, epoch+1)
		writer.add_image('streak image', out_streak_gray, epoch+1)

		torch.save(model.state_dict(), 'logs/real/latest.pth')


if __name__ == '__main__':
	start = timeit.default_timer()
	main()
	stop = timeit.default_timer()
	print('Total run time (min): ', (stop - start)/60)  