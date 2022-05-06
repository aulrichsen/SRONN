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

"""
Current State of the art
...
"""

def eval(model, val_dl, disp_imgs=False, log_img=False, table_type="validation"):
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
    
    if log_img:
        # Create a wandb Table to log images
        table = wandb.Table(columns=["low res", "pred", "high res", "PSNR", "SSIM"])
    
    with torch.no_grad():
        predicted_output = []
        X, Y = [], []
        for x_val, y_val in iter(val_dl):
            predicted_output.append(model(x_val))
            X.append(x_val)
            Y.append(y_val)
        predicted_output = torch.cat(predicted_output, dim=0)
        X = torch.cat(X, dim=0)
        Y = torch.cat(Y, dim=0)

    test_size = predicted_output.size()[0]
    for i in range (0, test_size):
        predicted_output = torch.clamp(predicted_output, min=0, max=1) # Clip values above 1 and below 0

        predict = predicted_output.cpu()[i,:,:,:]
        predict = predict.permute(1, 2, 0)
        predict = predict.detach().numpy()
        
        grountruth = Y.cpu()[i,:,:,:]
        grountruth = grountruth.permute(1, 2, 0)
        grountruth = grountruth.detach().numpy()
        cos = torch.nn.CosineSimilarity(dim=0)
        m = cos(predicted_output.cpu()[i,:,:,:],Y.cpu()[i,:,:,:])
        mn = np.average(m.detach().numpy())
        sam = math.acos(mn)*180/math.pi
        all_psnr.append(psnr(grountruth, predict))
        all_ssim.append(ssim(grountruth, predict))
        all_sam.append(sam)
        if log_img and i <= int(X.shape[1]/10):
            # Use dict to select display images for specific datasets
            ofs = 0 if X.shape[0] < 100 else 1+(i+1)*10     # Offset value for selection of better display images (i.e. ones that are not primarily black)
            img = X[i+ofs, i*10].cpu().numpy()
            out = predicted_output[i, i*10].cpu().numpy()
            tar = Y[i+ofs, i*10].cpu().numpy()
            table.add_data(wandb.Image(img*255), wandb.Image(out*255), wandb.Image(tar*255), psnr(tar, out), ssim(tar, out))
        if disp_imgs:
            fig, axs = plt.subplots(2)
            fig.suptitle('PSNR = ' + str(psnr(predict,grountruth)) + ', SSIM = '+ str(ssim(predict, grountruth)) + ', SAM = ' + str(sam))
            axs[0].imshow(predict[:,:,2],cmap='gray')
            axs[1].imshow(grountruth[:,:,2],cmap='gray')
    
    if log_img:
        wandb.log({table_type+"_predictions":table}, commit=False)

    if disp_imgs: fig.savefig(disp_imgs)

    avg_psnr = sum(all_psnr)/len(all_psnr)
    avg_ssim = sum(all_ssim)/len(all_ssim)
    avg_sam = sum(all_sam)/len(all_sam)

    return avg_psnr, avg_ssim, avg_sam

def train(model, train_dl, val_dl, test_dl, opt, best_vals=(0,0,1000), jt=None):
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
    wb_group:   wandb group name for saving
    jt:         wandb job type name for saving
    """

    arg_str = "Args: " + ', '.join(f'{k}={v}' for k, v in vars(opt).items())
    print(arg_str)

    with open('training_info.txt', 'w') as f:
        for k, v in vars(opt).items():
            f.write(f'{k}: {v}\n')

    print("Starting training...")
    logging.basicConfig(filename='training.log', filemode='w', level=logging.DEBUG)    # filemode='w' resets logger on every run  

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')     # If multiple GPUs, this is primary GPU

    wandb.init(
            project="HSI Super Resolution",
            group=opt.model_name,
            job_type=jt,
            config={"num_params": opt.num_params}
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
        start_time = time.time()
        
        model.train()
    
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            break

        total_loss = []
        for x_train, y_train in iter(train_dl):

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
        val_psnr, val_ssim, val_sam = eval(model, val_dl, log_img=log_img)
        psnrs.append(val_psnr)
        ssims.append(val_ssim)
        sams.append(val_sam)

        epoch_summary = f"Epoch: {epoch+1} | Loss: {round(loss.item(), 7)} | PSNR: {round(val_psnr, 3)} | SSIM: {round(val_ssim, 3)} | SAM: {round(val_sam, 3)} | LR: {current_lr} | Epoch time: {round(time.time() - start_time, 2)}s"
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
            torch.save(model.state_dict(), model_name+"_best_PSNR.pth.tar")
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            torch.save(model.state_dict(), model_name+"_best_SSIM.pth.tar")
        if val_sam < best_sam:
            best_sam = val_sam
            torch.save(model.state_dict(), model_name+"_best_SAM.pth.tar")       

    # Save validation stats
    with open('training_info.txt', 'a') as f:
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
    fig.savefig("Training Plots.png")
    #plt.show()

    # Test on all checkpointed models
    save_models = ["SAM", "PSNR", "SSIM"]
    for save_model in save_models:
        model.load_state_dict(torch.load(model_name+f"_best_{save_model}.pth.tar"))
        _psnr, _ssim, _sam = eval(model, test_dl, log_img=save_model=="SSIM", table_type="test")    # wandb log SSIM test images
        stats_msg = f"best {save_model} model: PSNR: {round(_psnr, 3)} | SSIM: {round(_ssim, 3)} | SAM: {round(_sam, 3)}"
        print(stats_msg)
        with open('training_info.txt', 'a') as f:
            f.write(stats_msg + '\n')
        # Finnish on SSIM model stats for wandb saving

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

    model = get_model(opt, channels).to(device)

    psnrs, ssims, sams = train(model, train_dl, val_dl, test_dl, opt, jt=dataset_name)

