import os

import scipy
import numpy as np
import matplotlib.pyplot as plt

from Load_Data import hsi_normalize_full


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

hsi = scipy.io.loadmat(DATA_DIR + '/PaviaU.mat').get('paviaU') 

hsi = hsi_normalize_full(hsi)
for c in range(hsi.shape[2]):
    if c < 10 or c > 95 or c==15 or c==90:
        plt.figure(figsize=(9,9))
        plt.imshow(hsi[200:400,50:250,c])
        plt.title(f"Band: {c}")
        plt.show()