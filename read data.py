import cv2
import matplotlib.pyplot as plt
import skimage.data 
import numpy as np


def heatmap(img):
    if len(img.shape) == 2:
        h,w = img.shape
        heat = np.zeros((h,w,3)).astype('uint8')
        
        heat[:,:,:] = np.transpose(cv2.applyColorMap(img[:,:],cv2.COLORMAP_JET),(2,0,1))
    else:
        h,w,c = img.shape
        heat = np.zeros((c,h,w)).astype('uint8')       
        heat[:,:,:] = np.transpose(cv2.applyColorMap(img[:,:,0],cv2.COLORMAP_JET),(2,0,1))
    return heat

train_data_path = 'training/real_world.txt'  
mat_files = open(train_data_path,'r').readlines()
file_num = len(mat_files)
print(file_num)
idx = 1
file_name = mat_files[idx % file_num]
print(file_name)
gt_file = '.'+file_name.split(' ')[1][:-1]
img_file = file_name.split(' ')[0][1:]
print(img_file)
O = cv2.imread(img_file)
plt.imshow(O)
plt.show()

B = cv2.imread(gt_file)
plt.imshow(B)
plt.show()

M = np.clip((O-B).sum(axis=2),0,1).astype(np.float32)
plt.imshow(M)
plt.show()


mask = heatmap(np.clip(M*255,0,255).astype(np.float32))
plt.imshow(mask)
plt.show()
