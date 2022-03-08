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


from utils import *

from Training import train
from models import Three_Layer_ONN, Three_Layer_ONN_BN
from Load_Data import get_paiva_data


def acquisition_function(hyperparameters, sample_save_file, model_type):
    """
    optimise based on psnr
    """

    global_best_psnr = 0
    samples = loadSamples(filename=sample_save_file)
    if samples: 
        currentSample = samples[-1]["Sample"] + 1
        for samp in samples:
            if samp["Best PSNR"] > global_best_psnr:
                global_best_psnr = samp["Best PSNR"]
    else:
        currentSample = 1
        
    lr, accum_iter = hyperparameters
    lr, accum_iter = float(lr), int(accum_iter)  # convert from numpy types to normal types

    print(currentSample, [lr, accum_iter])

    best_local_psnr = 0

    for i in range(3):
        if model_type == "Regular":
            model = Three_Layer_ONN(in_channels=channels, out_channels=channels).to(device)
        elif model_type == "Batch_Norm":
            model = Three_Layer_ONN_BN(in_channels=channels, out_channels=channels).to(device)
        else:
            assert False, "Invalid model_type"

        psnrs, ssims, sams = train(model, x_train, y_train, x_val, y_val, save_name=model_type, lr=0.0001, lr_step=10000, stats_disp=1000, best_vals=(global_best_psnr,1,0))    # Don't bother saving ssim and sam models
  
        best_train_psnr = max(psnrs)

        if best_train_psnr > best_local_psnr:
            # Use best sample iteration stats
            best_local_psnr = best_train_psnr
            best_psnrs = psnrs
            best_ssims = ssims
            best_sams = sams

    sample = {"Sample": currentSample, "X": [lr, accum_iter], "Best PSNR": best_local_psnr, "PSNRs": best_psnrs, "SSIMs": best_ssims, "SAMs": best_sams}
    saveSample(sample, data_save_name=sample_save_file)

    print(f"Best iteration PSNR: {best_local_psnr} | Overall best PSNR: {global_best_psnr}")
    return -best_local_psnr    # Negative for minimisation (actually want to maximise)    

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)    

    SAVE_NAME = "ONN_Paiva_test"

    x_train, y_train, x_val, y_val, x_test, y_test = get_paiva_data()
    x_train, y_train, x_val, y_val, x_test, y_test = x_train.to(device), y_train.to(device), x_val.to(device), y_val.to(device), x_test.to(device), y_test.to(device)

    channels = x_train.shape[1]

    model = Three_Layer_ONN(in_channels=channels, out_channels=channels).to(device)

    stats_file = os.getcwd() + "/stats.json"

    global_best_acc = 0


    # Specifiy hyperparameter exploration ranges
    lrMin = 0.000000001     # Smallest learning rate
    lrMax = 0.01             # Largest learning rate
    ssMin = 1            # Smallest step size
    ssMax = 100             # Largest setp size
    
    bounds = np.array([[lrMin, lrMax], [ssMin, ssMax]])
    print("Bounds:", bounds.shape)

    # === Regular Model Optimisation ===
    # Use custom kernel and estimator to match previous example
    m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gpr = GaussianProcessRegressor(kernel=m52, alpha=13-4) #noise**2)

    r = gp_minimize(lambda x: acquisition_function(np.array(x), "Regular_model_samples.json", "Regular"), 
                bounds.tolist(),
                base_estimator=gpr,
                acq_func='EI',      # expected improvement
                xi=0.01,            # exploitation-exploration trade-off
                n_calls=12,         # number of iterations
                n_random_starts=5)  # initial samples are provided

    # === BN Model Optimisation ===
    # Use custom kernel and estimator to match previous example
    m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gpr = GaussianProcessRegressor(kernel=m52, alpha=13-4) #noise**2)

    r = gp_minimize(lambda x: acquisition_function(np.array(x), "BN_model_samples.json", "Batch_Norm"), 
                bounds.tolist(),
                base_estimator=gpr,
                acq_func='EI',      # expected improvement
                xi=0.01,            # exploitation-exploration trade-off
                n_calls=12,         # number of iterations
                n_random_starts=5)  # initial samples are provided