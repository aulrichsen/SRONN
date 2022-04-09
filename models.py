import torch
import torch.nn as nn
import torch.nn.functional as F

from Self_ONN import Operator_Layer

from fastonn import SelfONN2d

class SRCNN(nn.Module):
    """
    One of the first super resolution models
    """
    def __init__(self, channels):
        super(SRCNN, self).__init__()
        self.name = "SRCNN"

        self.conv_1 = nn.Conv2d(channels, 128, kernel_size=9, padding='same') # Saye valid padding in SRCNN repo ...
        nn.init.xavier_uniform_(self.conv_1.weight)  # Init with golrot uniform weights
        self.conv_2 = nn.Conv2d(128, 64, kernel_size=3, padding='same')
        nn.init.xavier_uniform_(self.conv_2.weight)  # Init with golrot uniform weights
        self.conv_3 = nn.Conv2d(64, channels, kernel_size=5, padding='same')    # Saye valid padding in SRCNN repo ...
        nn.init.xavier_uniform_(self.conv_3.weight)  # Init with golrot uniform weights
        
        self.relu = nn.ReLU()

        self.num_params = self.conv_1.weight.numel() + self.conv_1.bias.numel() + self.conv_2.weight.numel() + self.conv_2.bias.numel() + self.conv_3.weight.numel() + self.conv_3.bias.numel()

    def forward(self, x):
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = self.conv_3(x)
        return x

class SRONN(nn.Module):
    """
    ONN Version of SRCNN
    """
    def __init__(self, channels, q_order=3):
        super(SRONN, self).__init__()
        self.name = "SRONN"

        self.op_1 = SelfONN2d(channels, 128, 9, q=q_order, padding='same')
        self.op_2 = SelfONN2d(128, 64, 3, q=q_order, padding='same')
        self.op_3 = SelfONN2d(64, channels, 5, q=q_order, padding='same')

        self.tanh = nn.Tanh()

        self.num_params = self.op_1.weight.numel() + self.op_1.bias.numel() + self.op_2.weight.numel() + self.op_2.bias.numel() + self.op_3.weight.numel() + self.op_3.bias.numel()

    def forward(self, x):
        x = self.tanh(self.op_1(x))
        x = self.tanh(self.op_2(x))
        x = self.op_3(x)
        return x


class SRONN_AEP(nn.Module):
    """
    ONN Version of SRCNN with approximately the same number of parameters as SRCNN.
    Hidden layer sizes of 32 and 16 (4x smaller than SRCNN), in fact has slightly less total parameters if default q val 3 used
    """
    def __init__(self, channels, q_order=3):
        super(SRONN_AEP, self).__init__()
        self.name = "SRONN_AEP"

        self.op_1 = SelfONN2d(channels, 32, 9, q=q_order, padding='same')
        self.op_2 = SelfONN2d(32, 16, 3, q=q_order, padding='same')
        self.op_3 = SelfONN2d(16, channels, 5, q=q_order, padding='same')

        self.tanh = nn.Tanh()

        self.num_params = self.op_1.weight.numel() + self.op_1.bias.numel() + self.op_2.weight.numel() + self.op_2.bias.numel() + self.op_3.weight.numel() + self.op_3.bias.numel()

    def forward(self, x):
        x = self.tanh(self.op_1(x))
        x = self.tanh(self.op_2(x))
        x = self.op_3(x)
        return x


class SRONN_BN(nn.Module):
    """
    ONN Version of SRCNN with Batch Normalisation
    """
    def __init__(self, channels, q_order=3, act=nn.Tanh()):
        super(SRONN_BN, self).__init__()
        self.name = "SRONN_BN"

        self.op_1 = SelfONN2d(channels, 128, 9, q=q_order, padding='same')
        self.bn_1 = nn.BatchNorm2d(128)
        self.op_2 = SelfONN2d(128, 64, 3, q=q_order, padding='same')
        self.bn_2 = nn.BatchNorm2d(64)
        self.op_3 = SelfONN2d(64, channels, 5, q=q_order, padding='same')

        self.act = act

        self.num_params = self.op_1.weight.numel() + self.op_1.bias.numel() + self.op_2.weight.numel() + self.op_2.bias.numel() + self.op_3.weight.numel() + self.op_3.bias.numel() + self.bn_1.weight.numel() + self.bn_1.bias.numel() + self.bn_2.weight.numel() + self.bn_2.bias.numel()
        
    def forward(self, x):
        x = self.act(self.bn_1(self.op_1(x)))
        x = self.act(self.bn_2(self.op_2(x)))
        x = self.op_3(x)
        return x

class SRONN_L2(nn.Module):
    """
    ONN Version of SRCNN with L2 Normalisation and no activation function.
    """
    def __init__(self, channels, q_order=3):
        super(SRONN_L2, self).__init__()
        self.name = "SRONN_L2"

        self.op_1 = SelfONN2d(channels, 128, 9, q=q_order, padding='same')
        self.op_2 = SelfONN2d(128, 64, 3, q=q_order, padding='same')
        self.op_3 = SelfONN2d(64, channels, 5, q=q_order, padding='same')
     
        self.num_params = self.op_1.weight.numel() + self.op_1.bias.numel() + self.op_2.weight.numel() + self.op_2.bias.numel() + self.op_3.weight.numel() + self.op_3.bias.numel()

    def forward(self, x):
        x = F.normalize(self.op_1(x), p=2, dim=(2,3))
        x = F.normalize(self.op_2(x), p=2, dim=(2,3))
        x = self.op_3(x)
        return x




# VDSRCNN - Very Deep Super Resolution CNN