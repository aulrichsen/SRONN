import argparse

import matplotlib.pyplot as plt

from utils import loadSamples

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
    parser.add_argument('--file', type=str, default='SRCNN_model_samples.json', help="File name of optimisation stats.")

    opt = parser.parse_args()
    #print_args(F)

    return opt

if __name__ == '__main__':
    opt = parse_opt()

    samples = loadSamples(opt.file)

    best_PSNRs = []
    best_sample = 1

    for samp in samples:
        if best_PSNRs:
            if samp["Best PSNR"] > max(best_PSNRs):
                best_sample = samp["Sample"]

        best_PSNRs.append(samp["Best PSNR"])

    print("Best PSNR on sample:", best_sample, "| PSNR:", samples[best_sample-1]["Best PSNR"])

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