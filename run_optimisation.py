from glob import glob
import numpy as np
import torch
import torch.nn as nn
import os

from sklearn.base import clone

import skopt
from skopt import gp_minimize
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern


from utils.general import *

from Training import train
from models import SRCNN, SRONN, SRONN_BN, SRONN_L2
from Load_Data import get_pavia_data


def acquisition_function(hyperparameters, sample_save_file, model_type):
    """
    optimise based on ssim
    """

    global_best_ssim = 0
    samples = loadSamples(filename=sample_save_file)
    if samples: 
        currentSample = samples[-1]["Sample"] + 1
        for samp in samples:
            if samp["Best SSIM"] > global_best_ssim:
                global_best_ssim = samp["Best SSIM"]
    else:
        currentSample = 1
        
    lr, lr_step = hyperparameters
    lr, lr_step = float(lr), int(lr_step)  # convert from numpy types to normal types

    print(currentSample, [lr, lr_step])

    best_local_ssim = 0

    for i in range(1):
        if model_type == "SRCNN":
            model = SRCNN(channels=channels).to(device)
        elif model_type == "SRONN":
            model = SRONN(channels=channels).to(device)
        elif model_type == "SRONN_L2":
            model = SRONN_L2(channels=channels).to(device)
        elif model_type == "SRONN_BN":
            model = SRONN_BN(channels=channels).to(device)
        elif model_type == "SRONN_BN_Relu":
            model = SRONN_BN(channels=channels, act=nn.ReLU()).to(device)
        else:
            assert False, "Invalid model_type"

        psnrs, ssims, sams = train(model, x_train, y_train, x_val, y_val, lr=lr, lr_step=lr_step, epochs=2000, stats_disp=10, best_vals=(100,global_best_ssim,0), wb_group=model.name)    # Don't bother saving ssim and sam models
  
        best_train_ssim = max(ssims)

        if best_train_ssim > best_local_ssim:
            # Use best sample iteration stats
            best_local_ssim = best_train_ssim
            best_psnrs = psnrs
            best_ssims = ssims
            best_sams = sams

    sample = {"Sample": currentSample, "X": [lr, lr_step], "Best SSIM": best_local_ssim, "PSNRs": best_psnrs, "SSIMs": best_ssims, "SAMs": best_sams}
    saveSample(sample, data_save_name=sample_save_file)

    print(f"Best iteration SSIM: {best_local_ssim} | Overall best SSIM: {global_best_ssim}")
    return -best_local_ssim    # Negative for minimisation (actually want to maximise)    

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)    

    x_train, y_train, x_val, y_val, x_test, y_test = get_pavia_data()
    x_train, y_train, x_val, y_val, x_test, y_test = x_train.to(device), y_train.to(device), x_val.to(device), y_val.to(device), x_test.to(device), y_test.to(device)

    channels = x_train.shape[1]

    # Specifiy hyperparameter exploration ranges
    lrMin = 0.000000001     # Smallest learning rate
    lrMax = 0.01             # Largest learning rate
    ssMin = 200            # Smallest step size
    ssMax = 2000             # Largest setp size
    
    bounds = np.array([[lrMin, lrMax], [ssMin, ssMax]])
    print("Bounds:", bounds.shape)


    n_its = 20
    n_rand_its = 7
    
    with open('optimisation_info.txt', 'w') as f:
            f.write(f'Bounds: {bounds}\n')
            f.write(f'Iterations: {n_its}, random: {n_rand_its}')

    # === Regular Model Optimisation ===
    def optimize(sample_savefile, model_type):
        # Use custom kernel and estimator to match previous example
        m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        gpr = GaussianProcessRegressor(kernel=m52, alpha=13-4) #noise**2)

        r = gp_minimize(lambda x: acquisition_function(np.array(x), sample_savefile, model_type), 
                    bounds.tolist(),
                    base_estimator=gpr,
                    acq_func='EI',      # expected improvement
                    xi=0.01,            # exploitation-exploration trade-off
                    n_calls=n_its,         # number of iterations
                    n_random_starts=n_rand_its)  # initial samples are provided

    #optimize("SRCNN_model_samples.json", model_type="SRCNN")

    #optimize("SRONN_model_samples.json", model_type="SRONN")

    optimize("SRONN_L2_model_samples.json", model_type="SRONN_L2")

    optimize("SRONN_BN_model_samples.json", model_type="SRONN_BN")
