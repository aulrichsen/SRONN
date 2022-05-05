import torch
import torch.nn as nn
import torch.nn.functional as F

#from Self_ONN import Operator_Layer
#from fastonn import SelfONN2d
from SelfONN import SelfONN2d

class SRCNN(nn.Module):
    """
    One of the first super resolution models
    """
    def __init__(self, channels, is_residual=False):
        super(SRCNN, self).__init__()
        self.name = "SRCNN"
        if is_residual: self.name += "_residual"
        self.is_residual = is_residual

        self.conv_1 = nn.Conv2d(channels, 128, kernel_size=9, padding='same') # Saye valid padding in SRCNN repo ...
        nn.init.xavier_uniform_(self.conv_1.weight)  # Init with golrot uniform weights
        self.conv_2 = nn.Conv2d(128, 64, kernel_size=3, padding='same')
        nn.init.xavier_uniform_(self.conv_2.weight)  # Init with golrot uniform weights
        self.conv_3 = nn.Conv2d(64, channels, kernel_size=5, padding='same')    # Saye valid padding in SRCNN repo ...
        nn.init.xavier_uniform_(self.conv_3.weight)  # Init with golrot uniform weights
        
        self.relu = nn.ReLU()

        self.num_params = self.conv_1.weight.numel() + self.conv_1.bias.numel() + self.conv_2.weight.numel() + self.conv_2.bias.numel() + self.conv_3.weight.numel() + self.conv_3.bias.numel()

    def forward(self, x):
        out = self.relu(self.conv_1(x))
        out = self.relu(self.conv_2(out))
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
    def __init__(self, channels, q=3, is_sig=False, is_residual=False):
        super(SRONN, self).__init__()
        self.name = "SRONN"
        if is_sig: self.name += "_sigmoid"
        self.is_sig = is_sig
        if is_residual: self.name += "_residual"
        self.is_residual = is_residual

        self.op_1 = SelfONN2d(channels, 128, 9, q=q, padding='same')
        self.op_2 = SelfONN2d(128, 64, 3, q=q, padding='same')
        self.op_3 = SelfONN2d(64, channels, 5, q=q, padding='same')

        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

        self.num_params = self.op_1.weight.numel() + self.op_1.bias.numel() + self.op_2.weight.numel() + self.op_2.bias.numel() + self.op_3.weight.numel() + self.op_3.bias.numel()

    def forward(self, x):
        out = self.tanh(self.op_1(x))
        out = self.tanh(self.op_2(out))
        out = self.op_3(out)

        if self.is_residual: out = torch.add(out, x)

        if self.is_sig: out = self.sig(out)

        return out


class SRONN_AEP(nn.Module):
    """
    ONN Version of SRCNN with approximately the same number of parameters as SRCNN.
    Hidden layer sizes of 32 and 16 (4x smaller than SRCNN), in fact has slightly less total parameters if default q val 3 used
    """
    def __init__(self, channels, q=3, is_residual=False):
        super(SRONN_AEP, self).__init__()
        self.name = "SRONN_AEP"
        if is_residual: self.name += "_residual"
        self.is_residual = is_residual

        self.op_1 = SelfONN2d(channels, 32, 9, q=q, padding='same')
        self.op_2 = SelfONN2d(32, 16, 3, q=q, padding='same')
        self.op_3 = SelfONN2d(16, channels, 5, q=q, padding='same')

        self.tanh = nn.Tanh()

        self.num_params = self.op_1.weight.numel() + self.op_1.bias.numel() + self.op_2.weight.numel() + self.op_2.bias.numel() + self.op_3.weight.numel() + self.op_3.bias.numel()

    def forward(self, x):
        out = self.tanh(self.op_1(x))
        out = self.tanh(self.op_2(out))
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



class SRONN_BN(nn.Module):
    """
    ONN Version of SRCNN with Batch Normalisation
    """
    def __init__(self, channels, q=3, act=nn.Tanh(), is_residual=False):
        super(SRONN_BN, self).__init__()
        self.name = "SRONN_BN"
        if is_residual: self.name += "_residual"
        self.is_residual = is_residual

        self.op_1 = SelfONN2d(channels, 128, 9, q=q, padding='same')
        self.bn_1 = nn.BatchNorm2d(128)
        self.op_2 = SelfONN2d(128, 64, 3, q=q, padding='same')
        self.bn_2 = nn.BatchNorm2d(64)
        self.op_3 = SelfONN2d(64, channels, 5, q=q, padding='same')

        self.act = act

        self.num_params = self.op_1.weight.numel() + self.op_1.bias.numel() + self.op_2.weight.numel() + self.op_2.bias.numel() + self.op_3.weight.numel() + self.op_3.bias.numel() + self.bn_1.weight.numel() + self.bn_1.bias.numel() + self.bn_2.weight.numel() + self.bn_2.bias.numel()
        
    def forward(self, x):
        x = self.act(self.bn_1(self.op_1(x)))
        x = self.act(self.bn_2(self.op_2(x)))
        x = self.op_3(x)

        if self.is_residual: out = torch.add(out, x)

        return x

class SRONN_L2(nn.Module):
    """
    ONN Version of SRCNN with L2 Normalisation and no activation function.
    """
    def __init__(self, channels, q=3):
        super(SRONN_L2, self).__init__()
        self.name = "SRONN_L2"

        self.op_1 = SelfONN2d(channels, 128, 9, q=q, padding='same')
        self.op_2 = SelfONN2d(128, 64, 3, q=q, padding='same')
        self.op_3 = SelfONN2d(64, channels, 5, q=q, padding='same')
     
        self.num_params = self.op_1.weight.numel() + self.op_1.bias.numel() + self.op_2.weight.numel() + self.op_2.bias.numel() + self.op_3.weight.numel() + self.op_3.bias.numel()

    def forward(self, x):
        x = F.normalize(self.op_1(x), p=2, dim=(2,3))
        x = F.normalize(self.op_2(x), p=2, dim=(2,3))
        x = self.op_3(x)
        return x




# VDSRCNN - Very Deep Super Resolution CNN