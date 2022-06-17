import math
import torch
import wandb
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from training_setup import get_disp_slices

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
            if b < X.shape[0] and c < X.shape[1]:
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
