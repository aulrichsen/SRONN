import argparse

import wandb
import torch
import torch.nn as nn

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
    parser.add_argument('--SISR', action='store_true', help='Perform SR on each channel individually.')

    opt = parser.parse_args()

    return opt

def get_optimizer(opt, model):
    if opt.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    elif opt.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    elif opt.optimizer == "RMSProp":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=opt.lr, alpha=0.99)
    #elif opt.optimizer == "FO_Adam":
    #    optimizer = Adam(model.parameters(), lr=opt.lr)
    else:
        assert False, "Invalid optimizer."

    return optimizer

def get_loss_function(opt):
    if opt.loss_function == "MSE":
        loss_function = nn.MSELoss()
    else:
        assert False, "Invalid loss function."

    return loss_function