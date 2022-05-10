import time
import logging
from datetime import datetime
from turtle import back

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
    #parser.add_argument('--distDP', action='store_true', help='Use Distributed Data Parallelism.')

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


    # Training start

    print("Starting training...")
    logging.basicConfig(filename='training.log', filemode='w', level=logging.DEBUG)    # filemode='w' resets logger on every run  

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')     # If multiple GPUs, this is primary GPU

    #wandb.init(
    #        project="HSI Super Resolution",
    #        group=model.name,
    #        job_type=jt,
    #        config={"num_params": model.num_params}
    #    )
    #wandb.config.update(opt)

    model_name = model.name         # If DaraParallel, can no longer access name attribute, save to variable

    gpus = torch.cuda.device_count()
    if gpus > 1:
        print(f"Using {gpus} GPUs")
        model = nn.DataParallel(model)
        model = model.to(device)

    opt.optimizer = "Adam"
    opt.lr = 0.000001
    optimizer = get_optimizer(opt, model)

    opt.lr_milestones = [2000]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.lr_milestones)

    opt.loss_function = "MSE"
    loss_function = get_loss_function(opt)

    opt.grad_clip = False
    opt.metrics_step = 10
    now = datetime.now()
    #wandb.run.name = now.strftime("%d/%m/%Y, %H:%M:%S")
    #print("wandb name:", wandb.run.name)

    best_psnr = 0
    best_ssim = 0
    best_sam = 100
    psnrs, ssims, sams = [], [], []

    for epoch in range(opt.epochs):
        epoch_start_time = time.time()
        
        model.train()
    
        forward_times = []
        backward_times = []
        opt_times = []

        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            break

        total_loss = []
        for x_train, y_train in train_dl:
            start_time = time.time()
            output = model(x_train)
            forward_times.append(time.time() - start_time)
            #print(x_train.shape, output.shape, y_train.shape)
            loss = loss_function(output, y_train)  
            total_loss.append(loss)
            start_time = time.time()
            loss.backward()
            backward_times.append(time.time()-start_time)

            if opt.grad_clip: nn.utils.clip_grad_norm_(model.parameters(), opt.clip_val/current_lr)

            start_time = time.time()
            optimizer.step()
            optimizer.zero_grad()
            opt_times.append(time.time()-start_time)
        
        scheduler.step()    # Reduce lr on every lr_step epochs
        
        loss = sum(total_loss)/len(total_loss)  # get average loss for the epoch for display

        log_img = False # (epoch + 1) % 200 == 0        # Only save images to wandb every 200 epcohs (speeds up sync)
        start_time = time.time()
        val_psnr, val_ssim, val_sam = eval(model, val_dl, log_img=log_img, disp_slices=get_disp_slices(opt.dataset, opt.SISR), SISR=opt.SISR)
        eval_time = time.time() - start_time
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
        #    metrics = {"train/loss": min(round(loss.item(),7), 0.01),   # Only log losses below 0.01 to avoid bad plot scale
        #                "val/PSNR": max(round(val_psnr, 5), 25),        # Only log PSNRs above 25 to avoid bad plot scale
        #                "val/SSIM": max(round(val_ssim, 5), 0.8),       # Only log SSIMs above 0.8 to avoid bad plot scale
        #                "val/SAM": min(round(val_sam, 5), 10)           # Only log SAMs below 10 to avoid bad plot scale
        #    }
        #    wandb.log(metrics)

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


        forward_times = [r(t) for t in forward_times]
        backward_times = [r(t) for t in backward_times]
        opt_times = [r(t) for t in opt_times]
        epoch_summary = f"Epoch: {epoch+1} | Forward: {forward_times} | Backward: {backward_times} | opt: {opt_times} | Eval: {r(eval_time)} | Epoch time: {r(time.time()-epoch_start_time)}s"
        print(epoch_summary)


    """
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
    """