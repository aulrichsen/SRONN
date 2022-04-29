import math
import logging
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import wandb
from datetime import datetime

from fastonn.utils.adam import Adam

from Load_Data import get_all_data, get_data, HSI_Dataset

from models import *

"""
Current State of the art



Draw.io for figures!

"""


def eval(model, val_dl, disp_imgs=False, log_img=False, table_type="validation"):
    """
    Get performance metrics
    PSNR: Peak Signal to Noise Ratio (higher the better); Spatial quality metric
    SSIM: Structural Similarity Index Measurement; Spatial quality metric
            Predicts perceived quality of videos. Simiilarity between 2 images. 1 if identical, lower = more difference.
    SAM:    Spectral Angle Mapper; Spectral quality metric, measures similarity between two vectors, pixel spectrum, averaged across each pixel
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            x_val, y_val = x_val.to(device), y_val.to(device)
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
        if log_img:
            img = X[i, i*10].cpu().numpy()
            out = predicted_output[i, i*10].cpu().numpy()
            tar = Y[i, i*10].cpu().numpy()
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    if opt.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    elif opt.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    elif opt.optimizer == "RMSProp":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=opt.lr, alpha=0.99)
    elif opt.optimizer == "FO_Adam":
        optimizer = Adam(model.parameters(), lr=opt.lr)
    else:
        assert False, "Invalid optimizer."

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.lr_milestones)

    if opt.loss_function == "MSE":
        lossFunction = nn.MSELoss()
    else:
        assert False, "Invalid loss function."

    wandb.init(
        project="HSI Super Resolution",
        group=model.name,
        job_type=jt,
        config={"num_params": model.num_params}
    )
    wandb.config.update(opt)


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
            x_train, y_train = x_train.to(device), y_train.to(device)
            output = model(x_train)
            #print(x_train.shape, output.shape, y_train.shape)
            loss = lossFunction(output, y_train)  
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
            #print(f"New best PSNR!: {round(psnr, 3)}, epoch {epoch+1}")
            torch.save(model.state_dict(), model.name+"_best_PSNR.pth.tar")
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            #print(f"New best SSIM!: {round(ssim, 3)}, epoch {epoch+1}")
            torch.save(model.state_dict(), model.name+"_best_SSIM.pth.tar")
        if val_sam < best_sam:
            best_sam = val_sam
            #print(f"New best SAM!: {round(sam, 3)}, epoch {epoch+1}")
            torch.save(model.state_dict(), model.name+"_best_SAM.pth.tar")       


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

    # Test best SAM model
    model.load_state_dict(torch.load(model.name+"_best_SAM.pth.tar"))
    _psnr, _ssim, _sam = eval(model, test_dl)
    stats_msg = f"best SAM model: PSNR: {round(_psnr, 3)} | SSIM: {round(_ssim, 3)} | SAM: {round(_sam, 3)}"
    print(stats_msg)
    with open('training_info.txt', 'a') as f:
        f.write(stats_msg + '\n')

    # Test best PSNR model
    model.load_state_dict(torch.load(model.name+"_best_PSNR.pth.tar"))
    _psnr, _ssim, _sam = eval(model, test_dl, disp_imgs="Best PSNR Out Image")
    stats_msg = f"best PSNR model: PSNR: {round(_psnr, 3)} | SSIM: {round(_ssim, 3)} | SAM: {round(_sam, 3)}"
    print(stats_msg)
    with open('training_info.txt', 'a') as f:
        f.write(stats_msg + '\n')

    # Test best SSIM model
    # Use best SSIM  model for wandb test metrics 
    model.load_state_dict(torch.load(model.name+"_best_SSIM.pth.tar"))
    _psnr, _ssim, _sam = eval(model, test_dl, log_img=True, table_type="test")
    stats_msg = f"best SSIM model: PSNR: {round(_psnr, 3)} | SSIM: {round(_ssim, 3)} | SAM: {round(_sam, 3)}"
    print(stats_msg)
    with open('training_info.txt', 'a') as f:
        f.write(stats_msg + '\n')

    wandb.run.summary["test_PSNR_from_best_SSIM_model"] = _psnr
    wandb.run.summary["test_SSIM_from_best_SSIM_model"] = _ssim
    wandb.run.summary["test_SAM_from_best_SSIM_model"] = _sam

    wandb.run.summary["best_val_PSNR"] = max(psnrs)
    wandb.run.summary["best_val_SSIM"] = max(ssims)
    wandb.run.summary["best_val_SAM"] = min(sams)

    wandb.finish()

    return psnrs, ssims, sams


def parse_train_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="SRONN", help='Model to use for training.')
    parser.add_argument('--q', type=int, default=3, help='q order of model. (for CNN is 1).')
    parser.add_argument('--dataset', type=str, default="Pavia", help="Dataset to train on.")
    parser.add_argument('--scale', type=int, default=2, help="Super resolution scale factor.")
    parser.add_argument('--epochs', type=int, default=10000, help="Number of epochs to train for.")
    parser.add_argument('--lr', type=float, default=0.0001, help="Starting training learning rate.")
    parser.add_argument('--lr_ms', dest="lr_milestones", nargs="+", type=int, default=[2500, 8000], help="Epochs at which to decrease learning rate by 10x.")
    parser.add_argument('--opt', dest="optimizer", type=str, default="Adam", help="Training optimizer to use.")
    parser.add_argument('--loss', dest="loss_function", type=str, default="MSE", help="Training loss function to use.")
    parser.add_argument('--ms', dest="metrics_step", type=int, default=10, help="Number of epochs between logging and display.")
    parser.add_argument('--clip', dest="grad_clip", action='store_true', help='Apply gradient clipping.')
    parser.add_argument('--clip_val', type=float, default=1, help='Gradient clipping parameter (Only applied if --clip set).')
    parser.add_argument('--trans', dest="weight_transfer", action='store_true', help='Transfer weights from SRCNN model.')
    parser.add_argument('--checkpoint', type=str, default="", help='Weight checkpoint to begin training with.')
    parser.add_argument('--SR_kernel', action='store_true', help='Use KernelGAN downsampling.')
    parser.add_argument('--bs', type=int, default=64, help="Training batch size.")

    opt = parser.parse_args()

    return opt

def get_ONN_weights(cnn_layer, onn_layer):
    onn_weight_shape = onn_layer.weight.shape

    w = torch.zeros(onn_weight_shape)
    cs = cnn_layer.weight.shape
    w[:cs[0], :cs[1], :cs[2], :cs[3]] = cnn_layer.weight

    return nn.Parameter(w)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)    

    opt = parse_train_opt()

    if opt.dataset == "All":
        train_data, val_data, test_data = get_all_data(res_ratio=opt.scale, SR_kernel=opt.SR_kernel)
        train_dl = DataLoader(train_data, batch_size=opt.bs, shuffle=True)
        val_dl = DataLoader(val_data, batch_size=opt.bs, shuffle=False)
        test_dl = DataLoader(test_data, batch_size=opt.bs, shuffle=False)
        dataset_name = "All x" + str(opt.scale)
        channels = 102
    else:
        x_train, y_train, x_val, y_val, x_test, y_test, dataset_name = get_data(dataset=opt.dataset, res_ratio=opt.scale, SR_kernel=opt.SR_kernel)
        x_train, y_train, x_val, y_val, x_test, y_test = x_train.to(device), y_train.to(device), x_val.to(device), y_val.to(device), x_test.to(device), y_test.to(device)

        train_data = HSI_Dataset(x_train, y_train)
        train_dl = DataLoader(train_data, batch_size=opt.bs, shuffle=True)
        val_data = HSI_Dataset(x_val, y_val)
        val_dl = DataLoader(val_data, batch_size=opt.bs, shuffle=False)
        test_data = HSI_Dataset(x_test, y_test)
        test_dl = DataLoader(test_data, batch_size=opt.bs, shuffle=False)

        channels = x_train.shape[1]

    if opt.model == "SRCNN":
        model = SRCNN(channels=channels).to(device)
        opt.q = 1
        opt.weight_transfer = False
    if opt.model == "SRCNN_residual":
        model = SRCNN(channels=channels, is_residual=True).to(device)
        opt.q = 1
        opt.weight_transfer = False
    if opt.model == "SRCNN_3D":
        model = SRCNN_3D().to(device)
        opt.q = 1
        opt.weight_transfer = False
    if opt.model == "SRCNN_3D_residual":
        model = SRCNN_3D(is_residual=True).to(device)
        opt.q = 1
        opt.weight_transfer = False
    elif opt.model == "SRONN":
        model = SRONN(channels=channels, q=opt.q).to(device)
    elif opt.model == "SRONN_residual":
        model = SRONN(channels=channels, q=opt.q, is_residual=True).to(device)
    elif opt.model == "SRONN_sigmoid_residual":
        model = SRONN(channels=channels, q=opt.q, is_sig=True, is_residual=True).to(device)
    elif opt.model == "SRONN_AEP":
        model = SRONN_AEP(channels=channels, q=opt.q).to(device)
        opt.weight_transfer = False
    elif opt.model == "SRONN_AEP_residual":
        model = SRONN_AEP(channels=channels, q=opt.q, is_residual=True).to(device)
        opt.weight_transfer = False
    elif opt.model == "SRONN_L2":
        model = SRONN_L2(channels=channels, q=opt.q).to(device)
    elif opt.model == "SRONN_BN":
        model = SRONN_BN(channels=channels, q=opt.q).to(device)
    elif opt.model == "WRF_ONN":
        model = WRF_ONN(channels=channels, q=opt.q).to(device)
    elif opt.model == "WRF_ONN_residual":
        model = WRF_ONN(channels=channels, q=opt.q, is_residual=True).to(device)
    else:
        assert False, "Invalid model type."
        
    if opt.weight_transfer:
        srcnn = SRCNN(channels=channels).to(device)
        srcnn.load_state_dict(torch.load("SRCNN_best_SSIM.pth.tar"))
        #srcnn.to(device)

        model.op_1.weight = get_ONN_weights(srcnn.conv_1, model.op_1)
        model.op_1.bias = srcnn.conv_1.bias
        model.op_2.weight = get_ONN_weights(srcnn.conv_2, model.op_2)
        model.op_2.bias = srcnn.conv_2.bias
        model.op_3.weight = get_ONN_weights(srcnn.conv_3, model.op_3)
        model.op_3.bias = srcnn.conv_3.bias

        model.to(device)

    if opt.checkpoint:
        model.load_state_dict(torch.load(opt.checkpoint))
        model.to(device)

    psnrs, ssims, sams = train(model, train_dl, val_dl, test_dl, opt, jt=dataset_name)

