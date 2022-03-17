import torch
import torch.nn as nn

from Self_ONN import Operator_Layer

from fastonn import SelfONN2d

class Single_Layer_Embedding_ONN(nn.Module):
    def __init__(self, in_channels, out_channels, q_order=3, emb_out=512, ks=3, act=nn.Tanh()):
        super(Single_Layer_Embedding_ONN, self).__init__()

        self.operator = Operator_Layer(in_channels, out_channels, ks, q_order)
        self.act = act  # Activation fiunction
        self.avg_pool = nn.AvgPool2d((1,1))
        self.embedding = nn.Linear(out_channels, emb_out)

    def forward(self, x):
        x = self.operator(x)
        x = self.act(x)
        x = self.avg_pool(x)
        x = self.embedding(x)
        x = nn.functional.normalize(x, p=2, dim=1)  # L2 norm
        return x


class Three_Layer_ONN(nn.Module):
    def __init__(self, in_channels, out_channels, q_order=3, act=nn.Tanh()):
        super(Three_Layer_ONN, self).__init__()

        self.OP_layer_1 = SelfONN2d(in_channels, 64, 9, q=q_order, padding='same')
        self.OP_layer_2 = SelfONN2d(64, 64, 3, q=q_order, padding='same')
        self.OP_layer_3 = SelfONN2d(64, out_channels, 5, q=q_order, padding='same')
        
        self.act = act  # Activation fiunction

    def forward(self, x):
        x = self.OP_layer_1(x)
        x = self.act(x)
        x = self.OP_layer_2(x)
        x = self.act(x)
        x = self.OP_layer_3(x)
        x = self.act(x)
        return x

class Three_Layer_ONN_BN(nn.Module):
    def __init__(self, in_channels, out_channels, q_order=3, act=nn.Tanh()):
        super(Three_Layer_ONN_BN, self).__init__()
        """
        Super Resolution Convolutional Neural Network (SRCNN)
        First CCN architecture for super resolution
        """

        self.OP_layer_1 = SelfONN2d(in_channels, 64, 9, q=q_order, padding='same')
        self.bn_1 = nn.BatchNorm2d(64)
        self.OP_layer_2 = SelfONN2d(64, 64, 3, q=q_order, padding='same')
        self.bn_2 = nn.BatchNorm2d(64)
        self.OP_layer_3 = SelfONN2d(64, out_channels, 5, q=q_order, padding='same')
        self.bn_3 = nn.BatchNorm2d(out_channels)
        
        self.act = act  # Activation fiunction

    def forward(self, x):
        x = self.OP_layer_1(x)
        x = self.bn_1(x)
        x = self.act(x)
        x = self.OP_layer_2(x)
        x = self.bn_2(x)
        x = self.act(x)
        x = self.OP_layer_3(x)
        x = self.bn_3(x)
        x = self.act(x)
        return x


class SRCNN(nn.Module):
    """
    One of the first super resolution models
    """
    def __init__(self, channels):
        super(SRCNN, self).__init__()

        self.conv_1 = nn.Conv2d(channels, 128, kernel_size=9, padding='same') # Saye valid padding in SRCNN repo ...
        nn.init.xavier_uniform_(self.conv_1.weight)  # Init with golrot uniform weights
        self.conv_2 = nn.Conv2d(128, 64, kernel_size=3, padding='same')
        nn.init.xavier_uniform_(self.conv_2.weight)  # Init with golrot uniform weights
        self.conv_3 = nn.Conv2d(64, channels, kernel_size=5, padding='same')    # Saye valid padding in SRCNN repo ...
        nn.init.xavier_uniform_(self.conv_3.weight)  # Init with golrot uniform weights
        
        self.relu = nn.ReLU()

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
        
        self.op_1 = SelfONN2d(channels, 128, 9, q=q_order, padding='same')
        self.op_2 = SelfONN2d(128, 64, 3, q=q_order, padding='same')
        self.op_3 = SelfONN2d(64, channels, 5, q=q_order, padding='same')
        
        self.init_weights()

        self.tanh = nn.Tanh()

    def init_weights(self):
        nn.init.xavier_uniform_(self.op_1.weight)  # Init with golrot uniform weights
        nn.init.xavier_uniform_(self.op_2.weight)  # Init with golrot uniform weights
        nn.init.xavier_uniform_(self.op_3.weight)  # Init with golrot uniform weights
        

    def forward(self, x):
        x = self.tanh(self.op_1(x))
        x = self.tanh(self.op_2(x))
        x = self.op_3(x)
        return x

class SRONN_BN(nn.Module):
    """
    ONN Version of SRCNN
    """
    def __init__(self, channels, q_order=3, act=nn.Tanh()):
        super(SRONN_BN, self).__init__()
        
        self.op_1 = SelfONN2d(channels, 128, 9, q=q_order, padding='same')
        nn.init.xavier_uniform_(self.op_1.weight)  # Init with golrot uniform weights
        self.bn_1 = nn.BatchNorm2d(128)
        self.op_2 = SelfONN2d(128, 64, 3, q=q_order, padding='same')
        nn.init.xavier_uniform_(self.op_2.weight)  # Init with golrot uniform weights
        self.bn_2 = nn.BatchNorm2d(64)
        self.op_3 = SelfONN2d(64, channels, 5, q=q_order, padding='same')
        nn.init.xavier_uniform_(self.op_3.weight)  # Init with golrot uniform weights
        
        self.act = act

    def forward(self, x):
        x = self.act(self.bn_1(self.op_1(x)))
        x = self.act(self.bn_2(self.op_2(x)))
        x = self.op_3(x)
        return x




# VDSRCNN - Very Deep Super Resolution CNN