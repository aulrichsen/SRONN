import torch
import torch.nn as nn

from Self_ONN import Operator_Layer

class Single_Layer_ONN(nn.Module):
    def __init__(self, in_channels, out_channels, q_order=3, emb_out=512, ks=3, act=nn.Tanh()):
        super(Single_Layer_ONN, self).__init__()

        self.operator = nn.Operator_Layer(in_channels, out_channels, ks, q_order)
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