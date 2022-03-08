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