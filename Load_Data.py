import torch
import torch.nn as nn
import  torchvision


from torch.utils.data import DataLoader, Dataset
#from SRDataset import SRDataset
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

import cv2
from patchify import patchify


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def hsi_normalize_full(hsi):
    tmp = hsi - np.min(hsi)
    img = tmp/np.max(tmp)
    return img

def bicubic_lr (img, ratio):
    # Function to create Low Resolution images using Bicubic Interpolation
    if ratio == 2:
        sigma = 0.8943
    else:
        sigma = 1.6986

    [h, w, d] = img.shape
    
    img = cv2.GaussianBlur(img, (0, 0), sigma, sigma) # run in a loop
    
    im_down = img[::ratio,::ratio, :]
    img_up = cv2.resize(im_down, (w, h),interpolation=cv2.INTER_CUBIC)
    return img_up

def get_paiva_data():
    #hsi =  scipy.io.loadmat('D:\HSI_SR_datasets\PaviaU.mat').get('paviaU') 
    hsi =  scipy.io.loadmat('C:/Users/psb15138/Documents/Uni/PTL/ONN/dataset/PaviaU.mat').get('paviaU') 

    print(hsi.shape)

    hsi = hsi_normalize_full(hsi)

    #hsi = np.float16(hsi)
    print(hsi.dtype)

    images = torch.tensor(hsi)

    images = images.permute(2,1,0)
    images = images.unsqueeze(1)
    print(images.shape)
    max_val = torch.max(images)
    min_val = torch.min(images)
    print("Max:", max_val)
    print("Min:", min_val)

    #images = images[30:40]

    #imshow(torchvision.utils.make_grid(images))

    # get image tiles (64x64 sub images of original hsi cube)
    hr_tiles = []
    patch_size = 64
    TILES = patchify(hsi, (patch_size,patch_size,hsi.shape[2]), step=patch_size )
    for i in range(TILES.shape[0]):
        for j in range(TILES.shape[1]):          
            hr_tiles.append(np.squeeze(TILES[i,j,:,:,:,:], axis = (0,)))

    # Normalize each tiles
    hr_tiles_nor = []
    for i in range(len(hr_tiles)):
        hr_tiles_nor.append(hsi_normalize_full(hr_tiles[i]))
        
    print(len(hr_tiles))
    print(hr_tiles_nor[0].shape)
    print(hr_tiles[-1].shape)


    # Create Low resolution tiles
    lr_tiles = []
    ratio = 2
    for i in list(range(len(hr_tiles))):
        #print("Processing tile # " , i)
        lr_tiles.append(bicubic_lr(hr_tiles_nor[i], ratio))


    X = np.array(lr_tiles)
    Y = np.array(hr_tiles_nor)
    X = np.moveaxis(X, -1, 1)
    Y = np.moveaxis(Y, -1, 1)
    X = np.float32(X)
    Y = np.float32(Y)

    (x_train, x_test, y_train, y_test) = train_test_split(X, Y, test_size=0.3,random_state=0)
    print('Training samples: ', x_train.shape[0])
    (x_val, x_test, y_val, y_test) = train_test_split(x_test, y_test, test_size=0.5,random_state=0)
    print('Validation samples: ', x_val.shape[0])
    print('Testing samples: ', x_test.shape[0])


    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)

    x_val = torch.from_numpy(x_val)
    y_val = torch.from_numpy(y_val)

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test