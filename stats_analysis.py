import os
import argparse

import torch
import torchvision
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from datetime import datetime

from models import *
from Load_Data import get_pavia_data
from Training import eval
from utils import loadSamples, imshow

"""
Stats file structure: list of dicts where each dict contains a givens sample's statistics
    Sample's statistics: 
        "Sample" : Sample index
        "X" : Hyperparameters selected
        "Best PSNR" : Best PSNR value of given iteration (across all X runs)
        "PSNRs" : Validation PSNR values across all epochs (for best run)
        "SSIMS" : Validation SSIM values across all epochs (for best run)
        "SAMs" : Validation SAM values across all epochs (for best run)
"""

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_file', type=str, default='SRCNN_model_samples.json', help="File name of optimisation stats.")
    parser.add_argument('--model_file', type=str, default='SRCNN_best_PSNR.pth.tar', help="File name of model.")
    parser.add_argument('--save_imgs', action='store_true', help='Save model outputs.')

    opt = parser.parse_args()
    #print_args(F)

    return opt

if __name__ == '__main__':
    opt = parse_opt()

    samples = loadSamples(opt.sample_file)

    best_PSNRs = []
    best_sample = 1

    for samp in samples:
        if best_PSNRs:
            if samp["Best PSNR"] > max(best_PSNRs):
                best_sample = samp["Sample"]

        best_PSNRs.append(samp["Best PSNR"])

    print("Best PSNR on sample:", best_sample, "| PSNR:", samples[best_sample-1]["Best PSNR"])
    print("Params:", samples[best_sample-1]["X"], samples[best_sample-1]["Sample"])

    #for samp in samples:
    #    print(samp["X"], samp["Best PSNR"])

    x = [i+1 for i in range(len(best_PSNRs))]
    plt.plot(x, best_PSNRs)
    plt.xlabel("Sample")
    plt.ylabel("PSNR")
    plt.title("Best PSNR Values per Sample")
    plt.show()

    best_sample_PSNRs = samples[best_sample-1]["PSNRs"]
    best_sample_SSIMs = samples[best_sample-1]["SSIMs"]
    best_sample_SAMs = samples[best_sample-1]["SAMs"]


    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle(f'Sample {best_sample} Validation Metric Plots')
    ax1.plot(best_sample_PSNRs)
    ax1.set_ylabel("PSNR")
    ax2.plot(best_sample_SSIMs)
    ax2.set_ylabel("SSIM")
    ax3.plot(best_sample_SAMs)
    ax3.set_ylabel("SAM")
    ax3.set_xlabel("Epoch")

    plt.show()


    if opt.save_imgs:
        _, _, _, _, test_X, test_Y = get_pavia_data()
        channels = test_X.shape[1]

        if "test" in opt.model_file:
            model_name = "Three_Layer_ONN"
            model = Three_Layer_ONN(in_channels=channels, out_channels=channels)
        elif "SRONN_BN" in opt.model_file:
            model_name = "SRONN_BN"
            model = SRONN_BN(channels=channels)
        elif "SRONN" in opt.model_file:
            model_name = "SRONN"
            model = SRONN(channels=channels)
        elif "SRCNN" in opt.model_file:
            model_name = "SRCNN"
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
        save_dir = "images/" + model_name + " " + now.strftime("%d/%m/%Y, %H:%M:%S")
        if not os.path.isdir(save_dir):
            os.mkdir("images/"+save_dir)
            

        with open(save_dir+'/info.txt', 'w') as f:
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
                SSIM = ssim(lab.permute(1,2,0).cpu().detach().numpy(), out.permute(1,2,0).cpu().detach().numpy())
                imshow(torchvision.utils.make_grid(images), title=save_dir+f"/Img {disp_img}, Slice {disp_chan}, PSNR {round(PSNR)}", plt_title=f"Low Res  | Output {round(PSNR, 2)} PSNR, {round(SSIM, 2)} |   High Res")




