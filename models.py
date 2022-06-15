import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools

#from Self_ONN import Operator_Layer
#from fastonn import SelfONN2d
from SelfONN import SelfONN2d
def get_model(opt, channels):

    norm_layer = get_norm_layer(opt.norm_type)

    if opt.model == "SRCNN":
        model = SRCNN(channels=channels, norm_layer=norm_layer, is_residual=opt.is_residual)
        opt.q = 1
        opt.weight_transfer = False
    elif opt.model == "SRONN":
        model = SRONN(channels=channels, q=opt.q, norm_layer=norm_layer, is_residual=opt.is_residual)
    elif opt.model == "SRONN_AEP":
        model = SRONN_AEP(channels=channels, q=opt.q, norm_layer=norm_layer, is_residual=opt.is_residual)
        opt.weight_transfer = False
    else:
        assert False, "Invalid model type."
        
    model_name = opt.model
    if opt.is_residual:
        model_name += "_residual"

    if opt.norm_type != "none":
        model_name += "_" + opt.norm_type + "_norm"

    init_weights(model, init_type=opt.init_type, init_gain=opt.init_gain)
    
    if opt.weight_transfer:
        srcnn = SRCNN(channels=channels) #.to(device)
        srcnn.load_state_dict(torch.load("SRCNN_best_SSIM.pth.tar"))
        #srcnn.to(device)

        model.op_1.weight = get_ONN_weights(srcnn.conv_1, model.op_1)
        model.op_1.bias = srcnn.conv_1.bias
        model.op_2.weight = get_ONN_weights(srcnn.conv_2, model.op_2)
        model.op_2.bias = srcnn.conv_2.bias
        model.op_3.weight = get_ONN_weights(srcnn.conv_3, model.op_3)
        model.op_3.bias = srcnn.conv_3.bias

        #model.to(device)

    if opt.checkpoint:
        model.load_state_dict(torch.load(opt.checkpoint))
        #model.to(device)

    return model, model_name


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight, 1.0, init_gain)
            init.constant_(m.bias, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


class Identity(nn.Module):
    def forward(self, x):
        return x

class L1_Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=1, dim=(2,3))

class L2_Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=(2,3))

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == "l1":
        def norm_layer(x): return L1_Norm()
    elif norm_type == "l2":
        def norm_layer(x): return L2_Norm()
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_ONN_weights(cnn_layer, onn_layer):
    onn_weight_shape = onn_layer.weight.shape

    w = torch.zeros(onn_weight_shape)
    cs = cnn_layer.weight.shape
    w[:cs[0], :cs[1], :cs[2], :cs[3]] = cnn_layer.weight

    return nn.Parameter(w)


class SRCNN(nn.Module):
    """
    One of the first super resolution models
    """
    def __init__(self, channels, is_residual=False, norm_layer=get_norm_layer('none')):
        super(SRCNN, self).__init__()
        self.name = "SRCNN"
        if is_residual: self.name += "_residual"
        self.is_residual = is_residual

        self.conv_1 = nn.Conv2d(channels, 128, kernel_size=9, padding='same') # Saye valid padding in SRCNN repo ...
        self.norm1 = norm_layer(128)
        #nn.init.xavier_uniform_(self.conv_1.weight)  # Init with golrot uniform weights
        self.conv_2 = nn.Conv2d(128, 64, kernel_size=3, padding='same')
        self.norm2 = norm_layer(64)
        #nn.init.xavier_uniform_(self.conv_2.weight)  # Init with golrot uniform weights
        self.conv_3 = nn.Conv2d(64, channels, kernel_size=5, padding='same')    # Saye valid padding in SRCNN repo ...
        #nn.init.xavier_uniform_(self.conv_3.weight)  # Init with golrot uniform weights
        
        self.relu = nn.ReLU()

        self.num_params = self.conv_1.weight.numel() + self.conv_1.bias.numel() + self.conv_2.weight.numel() + self.conv_2.bias.numel() + self.conv_3.weight.numel() + self.conv_3.bias.numel()

        if type(self.norm1) == nn.BatchNorm2d:
            self.num_params += self.norm1.weight.numel() + self.norm1.bias.numel() + self.norm2.weight.numel() + self.norm2.bias.numel()

    def forward(self, x):
        out = self.norm1(self.relu(self.conv_1(x)))
        out = self.norm2(self.relu(self.conv_2(out)))
        out = self.conv_3(out)

        if self.is_residual: out = torch.add(out, x)

        return out


class SRCNN_3D(nn.Module):
    """
    3D version of SRCNN 
    """
    def __init__(self, is_residual=False):
        super(SRCNN_3D, self).__init__()
        self.name = "SRCNN_3D"
        if is_residual: self.name += "_residual"
        self.is_residual = is_residual

        self.conv_1 = nn.Conv3d(1, 128, kernel_size=9, padding='same') # Saye valid padding in SRCNN repo ...
        nn.init.xavier_uniform_(self.conv_1.weight)  # Init with golrot uniform weights
        self.conv_2 = nn.Conv3d(128, 64, kernel_size=3, padding='same')
        nn.init.xavier_uniform_(self.conv_2.weight)  # Init with golrot uniform weights
        self.conv_3 = nn.Conv3d(64, 1, kernel_size=5, padding='same')    # Saye valid padding in SRCNN repo ...
        nn.init.xavier_uniform_(self.conv_3.weight)  # Init with golrot uniform weights
        
        self.relu = nn.ReLU()

        self.num_params = self.conv_1.weight.numel() + self.conv_1.bias.numel() + self.conv_2.weight.numel() + self.conv_2.bias.numel() + self.conv_3.weight.numel() + self.conv_3.bias.numel()

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.relu(self.conv_1(x))
        out = self.relu(self.conv_2(out))
        out = self.conv_3(out)

        if self.is_residual: out = torch.add(out, x)

        return out.squeeze()


class SRONN(nn.Module):
    """
    ONN Version of SRCNN
    """
    def __init__(self, channels, q=3, is_sig=False, is_residual=False, norm_layer=get_norm_layer('none')):
        super(SRONN, self).__init__()
        self.name = "SRONN"
        if is_sig: self.name += "_sigmoid"
        self.is_sig = is_sig
        if is_residual: self.name += "_residual"
        self.is_residual = is_residual

        self.op_1 = SelfONN2d(channels, 128, 9, q=q, padding='same')
        self.norm1 = norm_layer(128)
        self.op_2 = SelfONN2d(128, 64, 3, q=q, padding='same')
        self.norm2 = norm_layer(64)
        self.op_3 = SelfONN2d(64, channels, 5, q=q, padding='same')

        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

        self.num_params = self.op_1.weight.numel() + self.op_1.bias.numel() + self.op_2.weight.numel() + self.op_2.bias.numel() + self.op_3.weight.numel() + self.op_3.bias.numel()

        if type(self.norm1) == nn.BatchNorm2d:
            self.num_params += self.norm1.weight.numel() + self.norm1.bias.numel() + self.norm2.weight.numel() + self.norm2.bias.numel()

    def forward(self, x):
        out = self.norm1(self.tanh(self.op_1(x)))
        out = self.norm2(self.tanh(self.op_2(out)))
        out = self.op_3(out)

        if self.is_residual: out = torch.add(out, x)

        if self.is_sig: out = self.sig(out)

        return out


class SRONN_AEP(nn.Module):
    """
    ONN Version of SRCNN with approximately the same number of parameters as SRCNN.
    Hidden layer sizes of 32 and 16 (4x smaller than SRCNN), in fact has slightly less total parameters if default q val 3 used
    """
    def __init__(self, channels, q=3, is_residual=False, norm_layer=get_norm_layer('none')):
        super(SRONN_AEP, self).__init__()
        self.name = "SRONN_AEP"
        if is_residual: self.name += "_residual"
        self.is_residual = is_residual

        self.op_1 = SelfONN2d(channels, 32, 9, q=q, padding='same')
        self.norm1 = norm_layer(32)
        self.op_2 = SelfONN2d(32, 16, 3, q=q, padding='same')
        self.norm2 = norm_layer(32)
        self.op_3 = SelfONN2d(16, channels, 5, q=q, padding='same')

        self.tanh = nn.Tanh()

        self.num_params = self.op_1.weight.numel() + self.op_1.bias.numel() + self.op_2.weight.numel() + self.op_2.bias.numel() + self.op_3.weight.numel() + self.op_3.bias.numel()
        if type(self.norm1) == nn.BatchNorm2d:
            self.num_params += self.norm1.weight.numel() + self.norm1.bias.numel() + self.norm2.weight.numel() + self.norm2.bias.numel()


    def forward(self, x):
        out = self.norm1(self.tanh(self.op_1(x)))
        out = self.norm2(self.tanh(self.op_2(out)))
        out = self.op_3(out)

        if self.is_residual: out = torch.add(out, x)

        return out


class WRF_ONN(nn.Module):
    """
    ONN Version of SRCNN with approximately the same number of parameters as SRCNN.
    Hidden layer sizes of 32 and 16 (4x smaller than SRCNN), in fact has slightly less total parameters if default q val 3 used
    """
    def __init__(self, channels, q=3, is_residual=False):
        super(WRF_ONN, self).__init__()
        self.name = "WRF_ONN"
        if is_residual: self.name += "_residual"
        self.is_residual = is_residual

        self.op_1 = SelfONN2d(channels, 32, 19, q=q, padding='same')
        self.op_2 = SelfONN2d(32, 16, 7, q=q, padding='same')
        self.op_3 = SelfONN2d(16, channels, 11, q=q, padding='same')

        self.tanh = nn.Tanh()

        self.num_params = self.op_1.weight.numel() + self.op_1.bias.numel() + self.op_2.weight.numel() + self.op_2.bias.numel() + self.op_3.weight.numel() + self.op_3.bias.numel()

    def forward(self, x):
        out = self.tanh(self.op_1(x))
        out = self.tanh(self.op_2(out))
        out = self.op_3(out)

        if self.is_residual: out = torch.add(out, x)

        return out



# VDSRCNN - Very Deep Super Resolution CNN