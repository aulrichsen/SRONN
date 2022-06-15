import os
import math
import logging
import time

import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import wandb
from datetime import datetime

#from fastonn.utils.adam import Adam

from Load_Data import get_dataloaders
from training_setup import *
from models import get_model
from Test_Model import test_model

"""
Current State of the art
...
"""

def eval(model, val_dl, disp_imgs=False, log_img=False, table_type="validation", disp_slices=get_disp_slices(), SISR=False):
    """
    Get performance metrics
    PSNR: Peak Signal to Noise Ratio (higher the better); Spatial quality metric
    SSIM: Structural Similarity Index Measurement; Spatial quality metric
            Predicts perceived quality of videos. Simiilarity between 2 images. 1 if identical, lower = more difference.
    SAM:    Spectral Angle Mapper; Spectral quality metric, measures similarity between two vectors, pixel spectrum, averaged across each pixel
    """
    
    model.eval()

    all_psnr = []
    all_ssim = []
    all_sam = []
    
    with torch.no_grad():
        predicted_output = []
        X, Y = [], []
        for x_val, y_val in val_dl:
            predicted_output.append(model(x_val))
            X.append(x_val)
            Y.append(y_val)
        predicted_output = torch.cat(predicted_output, dim=0)
        X = torch.cat(X, dim=0)
        Y = torch.cat(Y, dim=0)

    predicted_output = torch.clamp(predicted_output, min=0, max=1) # Clip values above 1 and below 0
    cos = torch.nn.CosineSimilarity(dim=0)

    if SISR:
        # Reconstuct images for evaluation
        unique_tile_idxs = torch.unique(val_dl.tile_idxs)
        for idx in unique_tile_idxs:
            tile_idxs = torch.nonzero(torch.where(val_dl.tile_idxs==idx, 1, 0)).squeeze()
            predict = predicted_output[tile_idxs]
            ground_truth = Y[tile_idxs]
            m = cos(predict,ground_truth)
            mn = torch.mean(m)
            sam = math.acos(mn)*180/math.pi
            all_sam.append(sam)

            predict = predict.squeeze().permute(1, 2, 0)
            predict = predict.cpu().detach().numpy()
            ground_truth = ground_truth.squeeze().permute(1, 2, 0)
            ground_truth = ground_truth.cpu().detach().numpy()
            
            all_psnr.append(psnr(ground_truth, predict, data_range=1))
            all_ssim.append(ssim(ground_truth, predict, channel_axis=2))
    else:
        test_size = predicted_output.size()[0]
        for i in range(0, test_size):

            predict = predicted_output[i,:,:,:]
            ground_truth = Y[i,:,:,:]

            m = cos(predict,ground_truth)
            mn = torch.mean(m)
            sam = math.acos(mn)*180/math.pi
            all_sam.append(sam)
            
            predict = predict.permute(1, 2, 0)
            predict = predict.cpu().detach().numpy()
            ground_truth = ground_truth.permute(1, 2, 0)
            ground_truth = ground_truth.cpu().detach().numpy()
            
            all_psnr.append(psnr(ground_truth, predict, data_range=1))
            all_ssim.append(ssim(ground_truth, predict, channel_axis=2))
        
            if disp_imgs:
                fig, axs = plt.subplots(2)
                if SISR: sam = all_sam[val_dl.tile_idxs[i]]
                fig.suptitle('PSNR = ' + str(psnr(predict,ground_truth, data_range=1)) + ', SSIM = '+ str(ssim(predict, ground_truth, channel_axis=2)) + ', SAM = ' + str(sam))
                axs[0].imshow(predict[:,:,2],cmap='gray')
                axs[1].imshow(ground_truth[:,:,2],cmap='gray')
    
    if log_img:
        # Create a wandb Table to log images
        table = wandb.Table(columns=["low res", "pred", "high res", "PSNR", "SSIM"])
    
        for disp_slice in disp_slices:
            b, c = disp_slice["b"], disp_slice["c"]
            img = X[b, c].cpu().numpy()
            out = predicted_output[b, c].cpu().numpy()
            tar = Y[b, c].cpu().numpy()
            # ** Recorded table metrics are for since slice and NOT full HSI image. **
            table.add_data(wandb.Image(img*255), wandb.Image(out*255), wandb.Image(tar*255), psnr(tar, out), ssim(tar, out))    
        wandb.log({table_type+"_predictions":table}, commit=False)

    if disp_imgs: fig.savefig(disp_imgs)

    avg_psnr = sum(all_psnr)/len(all_psnr)
    avg_ssim = sum(all_ssim)/len(all_ssim)
    avg_sam = sum(all_sam)/len(all_sam)

    return avg_psnr, avg_ssim, avg_sam

def train(model, train_dl, val_dl, test_dl, dataset_name, opt, best_vals=(0,0,1000)):
    """
    train a model for HSI super resolution

    arguments
    model:      model architecture to be trained
    x_train:    training input data (low resolution data)
    y_train:    training labels (high resolution data)
    x_val:      validation input data (low resolution)
    y_val:      validation labels (high resolution)
    opt:        Training options:
        epochs:         number of epochs to train for
        lr:             initial training learning rate
        lr_milestones:  epoch milestones where learning rate reduction of lr*0.1 occurs
        metrics_step:   epoch divider to display training statistics on
    best_vals:  initial values for model saving to start on (psnr, ssim, sam), i.e. only save psnr on validation psnr of better than best_vals[0]
    """

    arg_str = "Args: " + ', '.join(f'{k}={v}' for k, v in vars(opt).items())
    print(arg_str)

    if not os.path.isdir("Results/"):
        os.mkdir("Results")

    now = datetime.now()
    save_dir = "Results/" + now.strftime("%d_%m_%Y %H_%M_%S") + " " + model.name + " " + dataset_name
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    with open(save_dir+'/training_info.txt', 'w') as f:
        for k, v in vars(opt).items():
            f.write(f'{k}: {v}\n')

    print("Starting training...")
    logging.basicConfig(filename=save_dir+'/training.log', filemode='w', level=logging.DEBUG)    # filemode='w' resets logger on every run  

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')     # If multiple GPUs, this is primary GPU

    if opt.wandb_group == 'none':
        opt.wandb_group = None
    elif opt.wandb_group == 'model_name':
        opt.wandb_group = opt.model_name

    if opt.wandb_jt == 'none':
        opt.wandb_jt = None
    elif opt.wandb_jt == 'dataset_name':
        opt.wandb_jt = dataset_name

    wandb.init(
            project="HSI-Super-Resolution",
            group=opt.wandb_group,
            job_type=opt.wandb_jt,
            config={"num_params": model.num_params}
        )
    wandb.config.update(opt)

    model_name = model.name         # If DaraParallel, can no longer access name attribute, save to variable

    gpus = torch.cuda.device_count()
    if gpus > 1:
        print(f"Using {gpus} GPUs")
        model = nn.DataParallel(model)
        model = model.to(device)

    optimizer = get_optimizer(opt, model)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.lr_milestones)

    loss_function = get_loss_function(opt)

    now = datetime.now()
    wandb.run.name = now.strftime("%d/%m/%Y, %H:%M:%S")
    print("wandb name:", wandb.run.name)

    best_psnr = best_vals[0]
    best_ssim = best_vals[1]
    best_sam = best_vals[2]
    psnrs, ssims, sams = [], [], []

    for epoch in range(opt.epochs):
        epoch_start_time = time.time()
        
        model.train()
    
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            break

        total_loss = []
        for x_train, y_train in train_dl:
            output = model(x_train)
            #print(x_train.shape, output.shape, y_train.shape)
            loss = loss_function(output, y_train)  
            total_loss.append(loss)
            loss.backward()

            if opt.grad_clip: nn.utils.clip_grad_norm_(model.parameters(), opt.clip_val/current_lr)

            optimizer.step()
            optimizer.zero_grad()
        
        scheduler.step()    # Reduce lr on every lr_step epochs
        
        loss = sum(total_loss)/len(total_loss)  # get average loss for the epoch for display

        log_img = (epoch + 1) % 200 == 0        # Only save images to wandb every 200 epcohs (speeds up sync)
        val_psnr, val_ssim, val_sam = eval(model, val_dl, log_img=log_img, disp_slices=get_disp_slices(opt.dataset, opt.SISR), SISR=opt.SISR)
        psnrs.append(val_psnr)
        ssims.append(val_ssim)
        sams.append(val_sam)

        epoch_summary = f"Epoch: {epoch+1} | Loss: {round(loss.item(), 7)} | PSNR: {round(val_psnr, 3)} | SSIM: {round(val_ssim, 3)} | SAM: {round(val_sam, 3)} | LR: {current_lr} | Epoch time: {round(time.time() - epoch_start_time, 2)}s"
        logging.info(epoch_summary)

        if (epoch + 1) % opt.metrics_step == 0 or epoch == 0:
            # Display and wandb log statistics            
            new_best_msg = ""
            if psnrs.index(max(psnrs)) >= len(psnrs) - opt.metrics_step: new_best_msg += " | new best PSNR! " + str(round(max(psnrs), 3))
            if ssims.index(max(ssims)) >= len(ssims) - opt.metrics_step: new_best_msg += " | new best SSIM! " + str(round(max(ssims), 3))
            if sams.index(min(sams)) >= len(sams) - opt.metrics_step: new_best_msg += " | new best SAM! " + str(round(min(sams), 3))
            
            print(epoch_summary + new_best_msg)
            metrics = {"train/loss": min(round(loss.item(),7), 0.01),   # Only log losses below 0.01 to avoid bad plot scale
                        "val/PSNR": max(round(val_psnr, 5), 25),        # Only log PSNRs above 25 to avoid bad plot scale
                        "val/SSIM": max(round(val_ssim, 5), 0.8),       # Only log SSIMs above 0.8 to avoid bad plot scale
                        "val/SAM": min(round(val_sam, 5), 10)           # Only log SAMs below 10 to avoid bad plot scale
            }
            wandb.log(metrics)
    
        # Check if new best made and save models if so
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), save_dir+"/"+model_name+"_best_PSNR.pth.tar")
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            torch.save(model.state_dict(), save_dir+"/"+model_name+"_best_SSIM.pth.tar")
        if val_sam < best_sam:
            best_sam = val_sam
            torch.save(model.state_dict(), save_dir+"/"+model_name+"_best_SAM.pth.tar")       

    # Save validation stats
    with open(save_dir+'/training_info.txt', 'a') as f:
        f.write("\nValidation Stats\n")
        f.write(f'Best PSNR: {best_psnr} | Best SSIM: {best_ssim} | Best SAM: {best_sam}\n')
        f.write("\nTest Stats\n")

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    ax[0].plot(psnrs)
    ax[0].set_title("PSNR Per Epoch")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("PSNR (dB)")
    ax[1].plot(ssims)
    ax[1].set_title("SSIM Per Epoch")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("SSIM")
    ax[2].plot(sams)
    ax[2].set_title("SAM Per Epoch")
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("SAM")
    fig.savefig(save_dir+"/Training Plots.png")
    #plt.show()

    # Test on all checkpointed models
    save_models = ["SAM", "PSNR", "SSIM"]
    for save_model in save_models:
        model.load_state_dict(torch.load(save_dir+"/"+model_name+f"_best_{save_model}.pth.tar"))
        _psnr, _ssim, _sam = eval(model, test_dl, log_img=save_model=="SSIM", table_type="test", disp_slices=get_disp_slices(opt.dataset, opt.SISR), SISR=opt.SISR)    # wandb log SSIM test images
        stats_msg = f"best {save_model} model: PSNR: {round(_psnr, 3)} | SSIM: {round(_ssim, 3)} | SAM: {round(_sam, 3)}"
        print(stats_msg)
        with open(save_dir+'/training_info.txt', 'a') as f:
            f.write(stats_msg + '\n')
        # Finnish on SSIM model stats for wandb saving

    disp_slices=None
    if opt.SISR:
        step = 40
        end = (test_dl.__len__() - 1) * opt.bs
        disp_slices = [{'b': i, 'c': 0} for i in range(0, end, step)]     # ** ADD SISR disp_slices here **
    test_model(model, test_dl, opt, save_dir=save_dir, disp_slices=disp_slices)

    wandb.run.summary["test_PSNR_from_best_SSIM_model"] = _psnr
    wandb.run.summary["test_SSIM_from_best_SSIM_model"] = _ssim
    wandb.run.summary["test_SAM_from_best_SSIM_model"] = _sam

    wandb.run.summary["best_val_PSNR"] = max(psnrs)
    wandb.run.summary["best_val_SSIM"] = max(ssims)
    wandb.run.summary["best_val_SAM"] = min(sams)

    wandb.finish()

    return psnrs, ssims, sams

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device:", device)    

    opt = parse_train_opt()

    train_dl, val_dl, test_dl, channels, dataset_name = get_dataloaders(opt, device)

    model, model_name = get_model(opt, channels).to(device)
    opt.model_name = model_name     # Update to add residual and norm type if applicable

    psnrs, ssims, sams = train(model, train_dl, val_dl, test_dl, dataset_name, opt)

