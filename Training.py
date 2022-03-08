import math
import copy
import logging

import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

from Load_Data import get_paiva_data

from models import Three_Layer_ONN

def eval(model, X, Y, disp_imgs=False):
    """
    Get performance metrics
    PSNR: Peak Signal to Noise Ratio (higher the better)
    SSIM: Structural Similarity Index Measure
            Predicts perceived quality of videos. Simiilarity between 2 images. 1 if identical, lower = more difference.
    SAM:    Unsure... Could be specral angle mapper?
    """

    model.eval()

    all_psnr = []
    all_ssim = []
    all_sam = []
    
    with torch.no_grad():
        predicted_output = model(X)
    test_size = predicted_output.size()[0]
    for i in range (0,test_size):
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
        all_psnr.append(psnr(predict,grountruth))
        all_ssim.append(ssim(predict, grountruth))
        all_sam.append(sam)
        if disp_imgs:
            fig, axs = plt.subplots(2)
            fig.suptitle('PSNR = ' + str(psnr(predict,grountruth)) + ', SSIM = '+ str(ssim(predict, grountruth)) + ', SAM = ' + str(sam))
            axs[0].imshow(predict[:,:,2],cmap='gray')
            axs[1].imshow(grountruth[:,:,2],cmap='gray')
    
    if disp_imgs: fig.savefig("HSI Output Images.png")

    avg_psnr = sum(all_psnr)/len(all_psnr)
    avg_ssim = sum(all_ssim)/len(all_ssim)
    avg_sam = sum(all_sam)/len(all_sam)

    return avg_psnr, avg_ssim, avg_sam

def train(model, x_train, y_train, x_val, y_val, save_name, lr=0.0001, lr_step=10000, stats_disp=10, best_vals=(0,0,1000)):
    """
    train a model for HSI super resolution

    arguments
    model:      model architecture to be trained
    x_train:    training input data (low resolution data)
    y_train:    training labels (high resolution data)
    x_val:      validation input data (low resolution)
    y_val:      validation labels (high resolution)
    save_name:  base filename for best models to be saved as
    lr:         initial training learning rate
    lr_step:    number of epochs between learning rate reduction of lr*0.1
    stats_disp: epoch divider to display training statistics on
    best_vals:  initial values for model saving to start on (psnr, ssim, sam), i.e. only save psnr on validation psnr of better than best_vals[0]
    """
    
    print("Starting training...")
    logging.basicConfig(filename='training.log', filemode='w', level=logging.DEBUG)    # filemode='w' resets logger on every run  

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.1)

    lossFunction = nn.MSELoss()

    best_psnr = best_vals[0]
    best_ssim = best_vals[1]
    best_sam = best_vals[2]
    psnrs, ssims, sams = [], [], []

    for epoch in range(10000):
        model.train()

        output = model(x_train)
        loss = lossFunction(output, y_train)  
        loss.backward()
    
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()    # Reduce lr on every lr_step epochs
        
        psnr, ssim, sam = eval(model, x_val, y_val)
        psnrs.append(psnr)
        ssims.append(ssim)
        sams.append(sam)
        
        epoch_summary = f"Epoch: {epoch+1}, | Loss: {round(loss.item(), 7)} | PSNR: {round(psnr, 3)} | SSIM: {round(ssim, 3)} | SAM: {round(sam, 3)}"
        logging.info(epoch_summary)

        if (epoch + 1) % stats_disp == 0 or epoch == 0:
            print(epoch_summary)
    
        # Check if new best made and save models if so
        if psnr > best_psnr:
            best_psnr = psnr
            #print(f"New best PSNR!: {round(psnr, 3)}, epoch {epoch+1}")
            torch.save(model.state_dict(), save_name+"_best_PSNR.pth.tar")
        if ssim > best_ssim:
            best_ssim = ssim
            #print(f"New best SSIM!: {round(ssim, 3)}, epoch {epoch+1}")
            torch.save(model.state_dict(), save_name+"_best_SSIM.pth.tar")
        if sam < best_sam:
            best_sam = sam
            #print(f"New best SAM!: {round(sam, 3)}, epoch {epoch+1}")
            torch.save(model.state_dict(), save_name+"_best_SAM.pth.tar")       

    return psnrs, ssims, sams

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)    

    SAVE_NAME = "ONN_Paiva_test"

    x_train, y_train, x_val, y_val, x_test, y_test = get_paiva_data()
    x_train, y_train, x_val, y_val, x_test, y_test = x_train.to(device), y_train.to(device), x_val.to(device), y_val.to(device), x_test.to(device), y_test.to(device)

    channels = x_train.shape[1]

    model = Three_Layer_ONN(in_channels=channels, out_channels=channels).to(device)
    
    psnrs, ssims, sams = train(model, x_train, y_train, x_val, y_val, SAVE_NAME)

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

    # Test best SSIM model
    model.load_state_dict(torch.load(SAVE_NAME+"_best_SSIM.pth.tar"))
    _psnr, _ssim, _sam = eval(model, x_test, y_test)
    print(f"best SSIM model: PSNR: {round(_psnr, 3)} | SSIM: {round(_ssim, 3)} | SAM: {round(_sam, 3)}")

    # Test best SAM model
    model.load_state_dict(torch.load(SAVE_NAME+"_best_SAM.pth.tar"))
    _psnr, _ssim, _sam = eval(model, x_test, y_test)
    print(f"best SAM model: PSNR: {round(_psnr, 3)} | SSIM: {round(_ssim, 3)} | SAM: {round(_sam, 3)}")

    # Test best PSNR model
    model.load_state_dict(torch.load(SAVE_NAME+"_best_PSNR.pth.tar"))
    _psnr, _ssim, _sam = eval(model, x_test, y_test, disp_imgs=True)
    print(f"best PSNR model: PSNR: {round(_psnr, 3)} | SSIM: {round(_ssim, 3)} | SAM: {round(_sam, 3)}")