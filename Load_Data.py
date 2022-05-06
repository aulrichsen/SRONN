import os
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import scipy.io
from scipy.ndimage import filters
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

def bicubic_lr(img, ratio):
    """
    Function to create Low Resolution images using Bicubic Interpolation
    """

    if ratio == 2:
        sigma = 0.8943
    else:
        sigma = 1.6986

    [h, w, d] = img.shape
    
    img = cv2.GaussianBlur(img, (0, 0), sigma, sigma) # run in a loop
    
    im_down = img[::ratio,::ratio, :]
    img_up = cv2.resize(im_down, (w, h),interpolation=cv2.INTER_CUBIC)
    return img_up

def get_data(dataset="Pavia", res_ratio=2, bands_to_remove=[], SR_kernel=False):

    # Find directory for relevant computer
    PC_DIR = 'C:/Users/psb15138/Documents/Uni/PTL/ONN/dataset'  # For my PC
    MAC_DIR = "../datasets"  # For my Mac
    NOUR_DIR = 'D:/HSI_SR_datasets'  # Nour's directory
    DIRS = [PC_DIR, MAC_DIR, NOUR_DIR]

    for data_dir in DIRS:
        if os.path.isdir(data_dir):
            DATA_DIR = data_dir
            break
    else:
        assert False, 'No data directory found. Please add path DIRS.'

    #hsi =  scipy.io.loadmat('D:\HSI_SR_datasets\PaviaU.mat').get('paviaU') 
    if dataset == "PaviaU":
        hsi = scipy.io.loadmat(DATA_DIR + '/PaviaU.mat')#.get('paviaU') 
        if res_ratio == 2:
            kernel = "LR_Kernels/PaviaU_kernel_x2.mat"
    elif dataset == "Botswana":
        hsi = scipy.io.loadmat(DATA_DIR + '/Botswana.mat')#.get('Botswana')
        if res_ratio == 2:
            kernel = "LR_Kernels/Botswana_kernel_x2.mat"
    elif dataset == 'Cuprite':
        hsi = scipy.io.loadmat(DATA_DIR + '/Cuprite_f970619t01p02_r02_sc03.a.rfl.mat')
        if res_ratio == 2:
            kernel = "LR_Kernels/Cuprite_kernel_x2.mat"
    elif dataset == 'Indian_Pines':
        hsi = scipy.io.loadmat(DATA_DIR + '/Indian_pines_corrected.mat')
        if res_ratio == 2:
            kernel = "LR_Kernels/Indian_pines_kernel_x2.mat"
    elif dataset == "KSC":
        hsi = scipy.io.loadmat(DATA_DIR + '/KSC.mat')
        if res_ratio == 2:
            kernel = "LR_Kernels/KSC_kernel_x2.mat"
    elif dataset == "Pavia":
        hsi = scipy.io.loadmat(DATA_DIR + '/Pavia.mat')
        if res_ratio == 2:
            kernel = "LR_Kernels/Pavia_kernel_x2.mat"
    elif dataset == "Salinas":
        hsi = scipy.io.loadmat(DATA_DIR + '/Salinas_corrected.mat')
        if res_ratio == 2:
            kernel = "LR_Kernels/Salinas_kernel_x2.mat"
    elif dataset == "Urban":
        hsi = scipy.io.loadmat(DATA_DIR + '/UrbanData.mat')
        if res_ratio == 2:
            kernel = "LR_Kernels/UrbanData_kernel_x2.mat"
    else:
        assert False, "Invalid dataset."
    dataset += " x"
    hsi = hsi.get(list(hsi.keys())[-1])     # Extract only data (Remove header etc)
    

    hsi = hsi_normalize_full(hsi)
    #hsi = np.float16(hsi)

    # ** Remove bands here **
    if bands_to_remove:
        bands_to_remove = [c if c >= 0 else hsi.shape[2]+c for c in bands_to_remove]    # Make any negative indexes their positive equivalent
        bands_to_remove.sort()
        bands_to_keep = [*range(hsi.shape[2])]
        for c in reversed(bands_to_remove):
            bands_to_keep.pop(c)

        hsi = hsi[:, :, bands_to_keep]

    images = torch.tensor(hsi)

    images = images.permute(2,1,0)
    images = images.unsqueeze(1)

    max_val = torch.max(images)
    min_val = torch.min(images)
    #print("Max:", max_val)
    #print("Min:", min_val)

    # get image tiles (64x64 sub images of original hsi cube)
    hr_tiles = []
    patch_size = 64
    TILES = patchify(hsi, (patch_size,patch_size,hsi.shape[2]), step=patch_size )
    for i in range(TILES.shape[0]):
        for j in range(TILES.shape[1]):          
            hr_tiles.append(np.squeeze(TILES[i,j,:,:,:,:], axis = (0,)))

    """
    # Normalize each tile
    hr_tiles_nor = []
    for hr_tile in hr_tiles:
        hr_tiles_nor.append(hsi_normalize_full(hr_tile))
    """

    # Create Low resolution tiles
    lr_tiles = []
    if SR_kernel:
        kernel = scipy.io.loadmat(kernel)
        for hr_tile_norm in hr_tiles:
            im_down = imresize(hr_tile_norm, scale_factor=1/res_ratio, kernel=kernel['Kernel'])
            w, h, c = hr_tile_norm.shape
            img_up = cv2.resize(im_down, (w, h), interpolation=cv2.INTER_CUBIC)
            lr_tiles.append(img_up)
    else:
        for hr_tile_norm in hr_tiles:
            #print("Processing tile # " , i)
            lr_tiles.append(bicubic_lr(hr_tile_norm, res_ratio))


    X = np.array(lr_tiles)
    Y = np.array(hr_tiles)
    X = np.moveaxis(X, -1, 1)
    Y = np.moveaxis(Y, -1, 1)
    X = np.float32(X)
    Y = np.float32(Y)

    (x_train, x_test, y_train, y_test) = train_test_split(X, Y, test_size=0.3,random_state=0)
    #print('Training samples: ', x_train.shape[0])
    (x_val, x_test, y_val, y_test) = train_test_split(x_test, y_test, test_size=0.5,random_state=0)
    #print('Validation samples: ', x_val.shape[0])
    #print('Testing samples: ', x_test.shape[0])


    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)

    x_val = torch.from_numpy(x_val)
    y_val = torch.from_numpy(y_val)

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    band_rm_str = ""
    if bands_to_remove:
        band_rm_str = " - "+ ', '.join(str(c) for c in bands_to_remove)

    dataset += str(res_ratio) + band_rm_str
    return x_train, y_train, x_val, y_val, x_test, y_test, dataset


class HSI_Dataset(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def imresize(im, scale_factor, kernel):
    # First standardize values and fill missing arguments (if needed) by deriving scale from output shape or vice versa
    #scale_factor, output_shape = fix_scale_and_size(im.shape, output_shape, scale_factor)
    scale_factor = [scale_factor, scale_factor, 1]  # scale x and y but not c

    output_shape = [int(im.shape[0] / scale_factor[0]), int(im.shape[1] / scale_factor[1]), im.shape[2]]

    return numeric_kernel(im, kernel, scale_factor, output_shape)

def numeric_kernel(im, kernel, scale_factor, output_shape):
   
    # First run a correlation (convolution with flipped kernel)
    out_im = np.zeros_like(im)
    for channel in range(im.shape[2]):
        out_im[:, :, channel] = filters.correlate(im[:, :, channel], kernel)

    # Then subsample and return
    return out_im[np.round(np.linspace(0, im.shape[0] - 1 / scale_factor[0], output_shape[0])).astype(int)[:, None],
           np.round(np.linspace(0, im.shape[1] - 1 / scale_factor[1], output_shape[1])).astype(int), :]


def get_all_data(res_ratio=2, SR_kernel=False, SISR=False):
    test_dataset = "PaviaU"

    datasets = ["Botswana", 'Cuprite', 'Indian_Pines', "KSC", "Pavia", "Salinas", "Urban"]

    C_len = 102     # Channel size of smallest dataset (Pavia)

    X_train, Y_train, X_val, Y_val, X_test, Y_test = [], [], [], [], [], []

    for dataset in datasets:
        _X_train, _Y_train, _X_val, _Y_val, _X_test, _Y_test, _ = get_data(dataset=dataset, res_ratio=res_ratio, SR_kernel=SR_kernel)

        #print(data_name, _X_train.shape[1])

        if SISR:
            X_train.append(_X_train.reshape(-1, 1, _X_train.shape[-2], _X_train.shape[-1]))
            Y_train.append(_Y_train.reshape(-1, 1, _Y_train.shape[-2], _Y_train.shape[-1]))
            X_val.append(_X_val.reshape(-1, 1, _X_val.shape[-2], _X_val.shape[-1]))
            Y_val.append(_Y_val.reshape(-1, 1, _Y_val.shape[-2], _Y_val.shape[-1]))
            X_test.append(_X_test.reshape(-1, 1, _X_test.shape[-2], _X_test.shape[-1]))
            Y_test.append(_Y_test.reshape(-1, 1, _Y_test.shape[-2], _Y_test.shape[-1]))
        else:
            num_splits = math.ceil(_X_train.shape[1]/C_len)

            X_train.append(_X_train[:, :C_len])
            Y_train.append(_Y_train[:, :C_len])
            X_val.append(_X_val[:, :C_len])
            Y_val.append(_Y_val[:, :C_len])
            X_test.append(_X_test[:, :C_len])
            Y_test.append(_Y_test[:, :C_len])
            if num_splits >= 2:
                X_train.append(_X_train[:, -C_len:])
                Y_train.append(_Y_train[:, -C_len:])
                X_val.append(_X_val[:, -C_len:])
                Y_val.append(_Y_val[:, -C_len:])
                X_test.append(_X_test[:, -C_len:])
                Y_test.append(_Y_test[:, -C_len:])
            if num_splits >= 3:
                c = int(_X_train.shape[1]/3)    # Start channel to select central bands 
                X_train.append(_X_train[:, c:c+C_len])
                Y_train.append(_Y_train[:, c:c+C_len])
                X_val.append(_X_val[:, c:c+C_len])
                Y_val.append(_Y_val[:, c:c+C_len])
                X_test.append(_X_test[:, c:c+C_len])
                Y_test.append(_Y_test[:, c:c+C_len])
        
    # Pavia U for testing
    X_1, Y_1, X_2, Y_2, X_3, Y_3, _ = get_data(dataset=test_dataset, res_ratio=res_ratio, SR_kernel=SR_kernel)
    X = torch.cat((X_1, X_2, X_3), dim=0)
    Y = torch.cat((Y_1, Y_2, Y_3), dim=0)
    if SISR:
        X_test.append(X.reshape(-1, 1, X.shape[-2], X.shape[-1]))
        Y_test.append(Y.reshape(-1, 1, Y.shape[-2], Y.shape[-1]))
    else:
        X_test.append(X[:, -C_len:])
        Y_test.append(Y[:, -C_len:])

    X_train = torch.cat(X_train, dim=0)
    Y_train = torch.cat(Y_train, dim=0)
    X_val = torch.cat(X_val, dim=0)
    Y_val = torch.cat(Y_val, dim=0)
    X_test = torch.cat(X_test, dim=0)
    Y_test = torch.cat(Y_test, dim=0)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def get_dataloaders(opt, device):
    if opt.dataset == "All":
        x_train, y_train, x_val, y_val, x_test, y_test = get_all_data(res_ratio=opt.scale, SR_kernel=opt.SR_kernel, SISR=opt.SISR)
        dataset_name = "All x" + str(opt.scale)
    else:
        x_train, y_train, x_val, y_val, x_test, y_test, dataset_name = get_data(dataset=opt.dataset, res_ratio=opt.scale, SR_kernel=opt.SR_kernel)
        if opt.SISR:
            x_train = x_train.reshape(-1, 1, x_train.shape[-2], x_train.shape[-1])
            y_train = y_train.reshape(-1, 1, x_train.shape[-2], x_train.shape[-1])
            x_val = x_val.reshape(-1, 1, x_val.shape[-2], x_val.shape[-1])
            y_val = y_val.reshape(-1, 1, y_val.shape[-2], y_val.shape[-1])
            x_test = x_test.reshape(-1, 1, x_test.shape[-2], x_test.shape[-1])
            y_test = y_test.reshape(-1, 1, y_test.shape[-2], y_test.shape[-1])

    train_data = TensorDataset(x_train.to(device), y_train.to(device))
    train_dl = DataLoader(train_data, batch_size=opt.bs, shuffle=True)
    val_data = TensorDataset(x_val.to(device), y_val.to(device))
    val_dl = DataLoader(val_data, batch_size=opt.bs*2, shuffle=False)
    test_data = TensorDataset(x_test.to(device), y_test.to(device))
    test_dl = DataLoader(test_data, batch_size=opt.bs*2, shuffle=False)

    channels = x_train.shape[1]

    return train_dl, val_dl, test_dl, channels, dataset_name


if __name__ == "__main__":
    
    train_data, val_data, test_data = get_all_data()

    print(train_data.X.shape, train_data.Y.shape)
    print(val_data.X.shape, val_data.Y.shape)
    print(test_data.X.shape, test_data.Y.shape)

    i=4
    X = val_data.X[i, i*10].cpu()
    Y = val_data.Y[i, i*10].cpu()

    print(X.shape, Y.shape)