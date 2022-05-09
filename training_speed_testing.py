import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models import SRONN
from Training import eval
from training_setup import *
from Load_Data import get_dataloaders

def r(num, dp=3):
    return round(num, dp)


def parse_speed_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--q', type=int, default=3, help='q order of model. (for CNN is 1).')
    parser.add_argument('--dataset', type=str, default="All", help="Dataset to train on.")
    parser.add_argument('--scale', type=int, default=2, help="Super resolution scale factor.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train for.")
    parser.add_argument('--SR_kernel', action='store_true', help='Use KernelGAN downsampling.')
    parser.add_argument('--bs', type=int, default=64, help="Training batch size.")
    parser.add_argument('--SISR', action='store_true', help='Perform SR on each channel individually.')
    parser.add_argument('--GPUs', type=int, default=1, help='Num GPUs to use.')
    parser.add_argument('--distDP', action='store_true', help='Use Distributed Data Parallelism.')

    opt = parser.parse_args()

    return opt

if __name__ == "__main__":
    opt = parse_speed_opt()
    arg_str = "Args: " + ', '.join(f'{k}={v}' for k, v in vars(opt).items())
    print(arg_str)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')     # If multiple GPUs, this is primary GPU

    start_time = time.time()
    train_dl, val_dl, test_dl, channels, dataset_name = get_dataloaders(opt, device)
    print("get_dataloaders time:", time.time()-start_time)
    print("channels:", channels)

    model = SRONN(channels=channels, q=opt.q, is_residual=True).to(device)

    model_name = model.name         # If DaraParallel, can no longer access name attribute, save to variable

    gpus = min(torch.cuda.device_count(), opt.GPUs)
    print(f"Using {gpus} GPUs")
    if gpus > 1:
        if opt.distDP:
            model = nn.parallel.DistributedDataParallel(model)
        else:
            model = nn.DataParallel(model)
        model = model.to(device)

    opt.optimizer = "Adam"
    opt.lr = 0.00001
    opt.loss_function = "MSE"
    optimizer = get_optimizer(opt, model)

    loss_function = get_loss_function(opt)

    for epoch in range(opt.epochs):
        epoch_start_time = time.time()

        model.train()

        forward_times = []
        backward_times = []
        opt_times = []

        total_loss = []
        for x_train, y_train in train_dl:
            start_time = time.time()
            output = model(x_train)
            forward_times.append(time.time()-start_time)
            #print(x_train.shape, output.shape, y_train.shape)
            loss = loss_function(output, y_train)  
            total_loss.append(loss)
            start_time = time.time()
            loss.backward()
            backward_times.append(time.time() - start_time)

            start_time = time.time()
            optimizer.step()
            optimizer.zero_grad()
            opt_times.append(time.time()-start_time)
        

        loss = sum(total_loss)/len(total_loss)  # get average loss for the epoch for display

        start_time = time.time()
        val_psnr, val_ssim, val_sam = eval(model, val_dl, log_img=False, disp_slices=get_disp_slices(opt.dataset, opt.SISR))
        eval_time = time.time()-start_time

        forward_times = [r(t) for t in forward_times]
        backward_times = [r(t) for t in backward_times]
        opt_times = [r(t) for t in opt_times]
        epoch_summary = f"Epoch: {epoch+1} | Forward: {forward_times} | Backward: {backward_times} | opt: {opt_times} | Eval: {r(eval_time)} | Epoch time: {r(time.time()-epoch_start_time)}s"
        print(epoch_summary)
    