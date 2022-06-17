import argparse

import wandb
import torch
import torch.nn as nn

def parse_train_opt():
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument('--model', type=str, default="SRONN", help='Model to use for training.')
    parser.add_argument('--q', type=int, default=3, help='q order of ONN model. (for CNN is 1).')
    parser.add_argument('--SISR', action='store_true', help='Perform SR on each channel individually.')
    parser.add_argument('--norm_type', type=str, default='none', help='Type of normalisation to use. none | batch | instance | l1 | l2.')
    parser.add_argument('--is_residual', action='store_true', help="Add residual connection to model.")
    parser.add_argument('--trans', dest="weight_transfer", action='store_true', help='Transfer weights from SRCNN model.')
    parser.add_argument('--checkpoint', type=str, default="", help='Weight checkpoint to begin training with.')
    parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')

    # Data parameters
    parser.add_argument('--dataset', type=str, default="PaviaU", help="Dataset to train on.")
    parser.add_argument('--scale', type=int, default=2, help="Super resolution scale factor.")
    parser.add_argument('--SR_kernel', action='store_true', help='Use KernelGAN downsampling.')
    parser.add_argument('--noise_var', type=float, default=0.00005, help='Variance of gaussian nosie added to dataset.')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50000, help="Number of epochs to train for.")
    parser.add_argument('--lr', type=float, default=0.0001, help="Starting training learning rate.")
    parser.add_argument('--lr_ms', dest="lr_milestones", nargs="+", type=int, default=[5000, 40000], help="Epochs at which to decrease learning rate by 10x.")
    parser.add_argument('--opt', dest="optimizer", type=str, default="Adam", help="Training optimizer to use.")
    parser.add_argument('--loss', dest="loss_function", type=str, default="MSE", help="Training loss function to use.")
    parser.add_argument('--ms', dest="metrics_step", type=int, default=100, help="Number of epochs between logging and display.")
    parser.add_argument('--clip', dest="grad_clip", action='store_true', help='Apply gradient clipping.')
    parser.add_argument('--clip_val', type=float, default=1, help='Gradient clipping parameter (Only applied if --clip set).')
    parser.add_argument('--bs', type=int, default=64, help="Training batch size.")

    # Logging parameters
    parser.add_argument('--wandb_group', type=str, default='model_name', help='name of wandb run group. model_name - use name of model | none - no group | other -custom.')
    parser.add_argument('--wandb_jt', type=str, default='dataset_name', help='name of wandb job_type. dataset_name - use name of dataset | none - no job_type | other -custom.')

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

def get_disp_slices(slices="default", SISR=False):
    """
    Choose specific indicies for wandb display

    ** NEED to make consistent **
    """
    
    if slices == "All":
        if SISR:
            # SISR, 102 channels
            disp_slices =  [{'b': 10656, 'c': 0}, {'b': 6772, 'c': 0}, {'b': 7790, 'c': 0}, {'b': 7302, 'c': 0}, {'b': 3908, 'c': 0}, {'b': 8391, 'c': 0}]
        else:
            disp_slices = [{'b': 94, 'c': 82}, {'b': 55, 'c': 0}, {'b': 126, 'c': 56}, {'b': 28, 'c': 98}, {'b': 88, 'c': 61}, {'b': 125, 'c': 74}]
    else:
        # default
        if SISR:
            disp_slices = [{'b': 0, 'c': 0}, {'b': 220, 'c': 0}, {'b': 330, 'c': 0}, {'b': 440, 'c': 0}, {'b': 550, 'c': 0}, {'b': 660, 'c': 0}]
        else: 
            disp_slices = [{'b': 0, 'c': 0}, {'b': 2, 'c': 20}, {'b': 3, 'c': 30}, {'b': 4, 'c': 40}, {'b': 5, 'c': 50}, {'b': 6, 'c': 60}]

    return disp_slices