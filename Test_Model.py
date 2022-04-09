import os
import argparse
from datetime import datetime

import cv2
import torch
import torchvision
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from models import *
from Load_Data import get_data
from Training import eval
from utils import imshow


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default='SRCNN_best_PSNR.pth.tar', help="File name of model.")
    parser.add_argument('--dataset', type=str, default="Pavia", help="Dataset trained on.")
    parser.add_argument('--scale', type=int, default=2, help="Super resolution scale factor.")
    
    opt = parser.parse_args()
    arg_str = "Args: " + ', '.join(f'{k}={v}' for k, v in vars(opt).items())
    print(arg_str)

    return opt

if __name__ == "__main__":
    
    opt = parse_opt()
    
    _, _, _, _, test_X, test_Y, _ = get_data(dataset=opt.dataset, res_ratio=opt.scale)
    channels = test_X.shape[1]

    if "SRONN_BN" in opt.model_file:
        model = SRONN_BN(channels=channels)
    elif "SRONN_AEP" in opt.model_file:
        model = SRONN_AEP(channels=channels)
    elif "SRONN" in opt.model_file:
        model = SRONN(channels=channels)
    elif "SRCNN" in opt.model_file:
        model = SRCNN(channels=channels)
    else:
        assert False, "Model file does not match any model in repo."
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    test_X = test_X.to(device)
    test_Y = test_Y.to(device)

    model.load_state_dict(torch.load(opt.model_file, map_location=torch.device(device)))
    model.to(device)
    model.eval()

    _psnr, _ssim, _sam = eval(model, test_X, test_Y)
    metric_info = f"PSNR: {round(_psnr, 3)} | SSIM: {round(_ssim, 3)} | SAM: {round(_sam, 3)}"
    print(metric_info)

    if not os.path.isdir("images/"):
        os.mkdir("images")

    now = datetime.now()
    save_dir = "images/" + model.name + " " + now.strftime("%d_%m_%Y %H_%M_%S")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        
    if not os.path.isdir(save_dir + "/objective"):
        os.mkdir(save_dir+'/objective')

    with open(save_dir+'/objective/info.txt', 'w') as f:
        f.write('Average Test Metrics\n')
        f.write(metric_info)

    with torch.no_grad(): 
        output = model(test_X)

    for disp_img in range(test_X.shape[0]):
        for disp_chan in range(disp_img, test_X.shape[1], 10):
            img = test_X[disp_img, disp_chan].unsqueeze(0).detach().cpu()
            out = output[disp_img, disp_chan].unsqueeze(0).detach().cpu()
            lab = test_Y[disp_img, disp_chan].unsqueeze(0).detach().cpu()

            out = torch.clamp(out, min=0, max=1)
            images = torch.stack([img, out, lab])

            PSNR = psnr(lab.permute(1,2,0).cpu().detach().numpy(), out.permute(1,2,0).cpu().detach().numpy())
            #SSIM = ssim(lab.permute(1,2,0).cpu().detach().numpy(), out.permute(1,2,0).cpu().detach().numpy())
            SSIM = ssim(lab[0].cpu().detach().numpy(), out[0].cpu().detach().numpy())
            imshow(torchvision.utils.make_grid(images), title=save_dir+'/objective'+f"/Img {disp_img}, Slice {disp_chan}, PSNR {round(PSNR)}", plt_title=f"Low Res  | Output {round(PSNR, 2)} PSNR, {round(SSIM, 2)} SSIM |   High Res")

    if not os.path.isdir(save_dir + "/true"):
        os.mkdir(save_dir + '/true')

    # Perform Super resolution on original data (subjective)
    for disp_img in range(test_Y.shape[0]):
        for disp_chan in range(disp_img, test_Y.shape[1], 10):
            img = test_Y[disp_img].detach().cpu()
            img = img.permute(1,2,0).numpy()
            [h, w, d] = img.shape
            
            test_X_true = cv2.resize(img, (w*opt.scale, h*opt.scale),interpolation=cv2.INTER_CUBIC)
            test_X_true = torch.Tensor(test_X_true).to(device).permute(2,0,1)
            test_X_true = test_X_true.unsqueeze(0)
            with torch.no_grad(): 
                output = model(test_X_true)

            out = torch.clamp(output[:,disp_chan], min=0, max=1)
            images = torch.stack([test_X_true[:, disp_chan], out])
            imshow(torchvision.utils.make_grid(images.cpu().detach()), title=save_dir+'/true'+f"/Img {disp_img}, Slice {disp_chan}", plt_title=f"Interpolation | Output")
