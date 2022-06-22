import os
import argparse
from datetime import datetime

import cv2
import torch
from torch.utils.data import DataLoader
import torchvision
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from models import *
from Load_Data import get_data, HSI_Dataset
from eval import eval
from utils.general import imshow


def parse_opt():
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument('--model', type=str, default="SRONN", help='Model to used in training for evaluation.')
    parser.add_argument('--q', type=int, default=3, help='q order of ONN model. (for CNN is 1).')
    parser.add_argument('--SISR', action='store_true', help='Perform SR on each channel individually.')
    parser.add_argument('--norm_type', type=str, default='none', help='Type of normalisation to use. none | batch | instance | l1 | l2.')
    parser.add_argument('--is_residual', action='store_true', help="Add residual connection to model.")
    parser.add_argument('--trans', dest="weight_transfer", action='store_true', help='Transfer weights from SRCNN model.')
    parser.add_argument('--checkpoint', type=str, default='SRCNN_best_PSNR.pth.tar', help="File name of model weights to load in.")
    parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')

    # Dataset parameters    
    parser.add_argument('--dataset', type=str, default="Pavia", help="Dataset trained on.")
    parser.add_argument('--scale', type=int, default=2, help="Super resolution scale factor.")
    parser.add_argument('--SR_kernel', action='store_true', help='Use KernelGAN downsampling.')
    parser.add_argument('--noise_var', type=float, default=0.00005, help='Variance of gaussian nosie added to dataset.')

    opt = parser.parse_args()
    arg_str = "Args: " + ', '.join(f'{k}={v}' for k, v in vars(opt).items())
    print(arg_str)

    return opt


def test_model(model, test_dl, opt, save_dir=None, disp_slices=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()    
    _psnr, _ssim, _sam = eval(model, test_dl)
    metric_info = f"PSNR: {round(_psnr, 3)} | SSIM: {round(_ssim, 3)} | SAM: {round(_sam, 3)}"
    print(metric_info)

    if not os.path.isdir("Results/"):
        os.mkdir("Results")

    now = datetime.now()
    
    if save_dir is None:
        # Default saving directory
        save_dir = "Results/" + now.strftime("%d_%m_%Y %H_%M_%S") + " " + model.name
    elif save_dir[:8] != "Results/":
        save_dir = "Results/" + save_dir    

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        
    if not os.path.isdir(save_dir + "/objective"):
        os.mkdir(save_dir+'/objective')

    with open(save_dir+'/objective/info.txt', 'w') as f:
        f.write('Average Test Metrics\n')
        f.write(metric_info)

    with torch.no_grad(): 
        output = []
        test_X = []
        test_Y = []
        for x_test, y_test in iter(test_dl):
            output.append(model(x_test))
            test_X.append(x_test)
            test_Y.append(y_test)

        output = torch.cat(output, dim=0)
        test_X = torch.cat(test_X, dim=0)
        test_Y = torch.cat(test_Y, dim=0)

    if disp_slices is None:
        # Default display slices
        disp_slices = []
        for disp_img in range(test_Y.shape[0]):
            for disp_chan in range(disp_img, test_Y.shape[1], 10):
                disp_slices.append({'b': disp_img, 'c': disp_chan})

    for disp_slice in disp_slices:
        img = test_X[disp_slice['b'], disp_slice['c'], :, :].unsqueeze(0).detach().cpu()
        out = output[disp_slice['b'], disp_slice['c'], :, :].unsqueeze(0).detach().cpu()
        lab = test_Y[disp_slice['b'], disp_slice['c'], :, :].unsqueeze(0).detach().cpu()

        out = torch.clamp(out, min=0, max=1)
        images = torch.stack([img, out, lab])

        PSNR = psnr(lab.permute(1,2,0).cpu().detach().numpy(), out.permute(1,2,0).cpu().detach().numpy())
        #SSIM = ssim(lab.permute(1,2,0).cpu().detach().numpy(), out.permute(1,2,0).cpu().detach().numpy())
        SSIM = ssim(lab[0].cpu().detach().numpy(), out[0].cpu().detach().numpy())
        imshow(torchvision.utils.make_grid(images), title=save_dir+'/objective'+f"/Img {disp_slice['b']}, Slice {disp_slice['c']}, PSNR {round(PSNR)}", plt_title=f"Low Res  | Output {round(PSNR, 2)} PSNR, {round(SSIM, 2)} SSIM |   High Res")

    if not os.path.isdir(save_dir + "/true"):
        os.mkdir(save_dir + '/true')


    # Perform Super resolution on original data (subjective)
    for disp_slice in disp_slices:
        if opt.SISR:
            img = test_Y[disp_slice['b'], disp_slice['c'], :, :].detach().cpu().numpy()
            [h, w] = img.shape
        else:
            img = test_Y[disp_slice['b'], :, :, :].detach().cpu()
            img = img.permute(1,2,0).numpy()
            [h, w, d] = img.shape
            
        test_X_true = cv2.resize(img, (w*opt.scale, h*opt.scale), interpolation=cv2.INTER_CUBIC)
        test_X_true = torch.Tensor(test_X_true).to(device)
        test_X_true = test_X_true.unsqueeze(0)
        if opt.SISR:
            test_X_true = test_X_true.unsqueeze(0)
        else:
            test_X_true = test_X_true.permute(0,3,1,2)
        with torch.no_grad(): 
            output = model(test_X_true)

        out = torch.clamp(output[:, disp_slice['c']], min=0, max=1)
        images = torch.stack([test_X_true[:, disp_slice['c']], out])
        imshow(torchvision.utils.make_grid(images.cpu().detach()), title=save_dir+'/true'+f"/Img {disp_slice['b']}, Slice {disp_slice['c']}", plt_title=f"Interpolation | Output")


if __name__ == "__main__":
    
    opt = parse_opt()
    
    _, _, _, _, test_X, test_Y, _ = get_data(dataset=opt.dataset, res_ratio=opt.scale, SR_kernel=opt.SR_kernel)
    channels = test_X.shape[1]

    model, _ = get_model(opt, channels)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    test_X = test_X.to(device)
    test_Y = test_Y.to(device)
    
    bs= 4
    test_data = HSI_Dataset(test_X, test_Y)
    test_dl = DataLoader(test_data, batch_size=bs, shuffle=False)

    model.to(device)
    
    test_model(model, test_dl, opt)