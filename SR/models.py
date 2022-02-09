import torch
import torch.nn as nn
from torch import add

from Self_ONN import Operator_Layer

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
    def __init__(self, in_channels, out_channels, q_order=3, ks=3, act=nn.Tanh()):
        super(Three_Layer_ONN, self).__init__()

    #     self.OP_layer_1 = Operator_Layer(in_channels, 64, ks, q_order)
    #     self.OP_layer_2 = Operator_Layer(64, 32, ks, q_order)
    #     self.OP_layer_3 = Operator_Layer(32, out_channels, ks, q_order)
        
    #     self.act = act  # Activation fiunction

    # def forward(self, x):
    #     x = self.OP_layer_1(x)
    #     x = self.act(x)
    #     x = self.OP_layer_2(x)
    #     x = self.act(x)
    #     x = self.OP_layer_3(x)
    #     x = self.act(x)
    #     return x
        self.layer_1 = Operator_Layer(in_channels, 32, ks, q_order)
        self.layer_2 = Operator_Layer(32, 128, 1, q_order)
        self.layer_3 = Operator_Layer(128, 32, ks, q_order)
        self.layer_4 = Operator_Layer(32, 48, ks, q_order)
        self.layer_5 = Operator_Layer(48, 48, 5, q_order)
        self.layer_6 = Operator_Layer(48, out_channels, 5, q_order)
        
        self.act = act  # Activation fiunction
        
        self.ADD = add

    def forward(self, x):
        x1 = self.OP_layer_1(x)
        x1 = self.act(x1)
        x2 = self.OP_layer_2(x1)
        x2 = self.act(x2)
        x3 = self.OP_layer_3(x2)
        x3 = self.act(x3)
        x4 = self.ADD(x1,x3)
        x5 = self.OP_layer_4(x4)
        x5 = self.act(x5)
        x6 = self.OP_layer_5(x5)
        x6 = self.act(x6)
        x7 = self.ADD(x5,x6)
        x8 = self.OP_layer_4(x7)
        x8 = self.act(x8)
    
    


# class WDSR_ONN(nn.Module):
#     def __init__(self, in_channels, out_channels, q_order=3, ks=3, act=nn.Tanh()):
#         super(WDSR_ONN, self).__init__()

#         self.layer_1 = Operator_Layer(in_channels, 32, ks, q_order)
#         self.layer_2 = Operator_Layer(32, 128, 1, q_order)
#         self.layer_3 = Operator_Layer(128, 32, ks, q_order)
#         self.layer_4 = Operator_Layer(32, 48, ks, q_order)
#         self.layer_5 = Operator_Layer(48, 48, 5, q_order)
#         self.layer_6 = Operator_Layer(48, out_channels, 5, q_order)
        
#         self.act = act  # Activation fiunction
        
#         self.ADD = add

#     def forward(self, x):
#         x1 = self.OP_layer_1(x)
#         x1 = self.act(x1)
#         x2 = self.OP_layer_2(x1)
#         x2 = self.act(x2)
#         x3 = self.OP_layer_3(x2)
#         x3 = self.act(x3)
#         x4 = self.ADD(x1,x3)
#         x5 = self.OP_layer_4(x4)
#         x5 = self.act(x5)
#         x6 = self.OP_layer_5(x5)
#         x6 = self.act(x6)
#         x7 = self.ADD(x5,x6)
#         x8 = self.OP_layer_4(x7)
#         x8 = self.act(x8)
#         return x8   