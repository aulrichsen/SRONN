"""
24/09/2021
author: Alexander Ulrichsen
Alexander Ulrichsen's implementation Self ONNs from the paper "Self-organized Operational Neural Networks with Generative Neurons"
link: https://www.sciencedirect.com/science/article/pii/S0893608021000782

Summation pool operator assumed

**Note** Look into Chebychev polynomials to replace Taylor/MacLaurin series - better approximation apparently
"""

import torch
import torch.nn as nn


class Operator_Layer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, ks: int = 3, q_order: int = 3):
        """
        q_order:        the MacLaurin series approximation order    
        in_channels:    number of image channels layer input has
        out_channels:   number of desired output channels from layer
        ks:             convlution kernel filter size
        """
        super(Operator_Layer, self).__init__()

        self.in_channels = in_channels
        self.q_order = q_order

        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

        self.operator = nn.Conv3d(in_channels*q_order, out_channels, ks, padding='same')

    def forward(self, x):
        # Raise inputs up to power q 
        power_input = [x]
        for q in range(2, self.q_order+1):
            power_input.append(torch.pow(x, q))
        power_input = torch.cat(power_input, 1)

        operator_out = self.operator(power_input)

        return operator_out

class Operator_Layer_Split(nn.Module):
    """
    Operator Layer where each q component is a seperate group so that learning rates can be set individually
    """

    def __init__(self, in_channels, out_channels, ks: int = 3, q_order: int = 3):
        """
        q_order: the MacLaurin series order approximation
        """
        super(Operator_Layer_Split, self).__init__()

        self.in_channels = in_channels
        self.q_order = q_order

        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

        self.operators = nn.ModuleList([nn.Conv2d(in_channels, out_channels, ks)] + [nn.Conv2d(in_channels, out_channels, ks, bias=False) for _ in range(q_order-1)]    )
        #self.bias = nn.Parameter(torch.rand(1,out_channels,1,1))

    def forward(self, x):
        # Raise inputs up to power q 
        out = self.operators[0](x)
        for q in range(2, self.q_order+1):
            out += self.operators[q-1](torch.pow(x, q))

        return out

if __name__ == "__main__":
    ip = torch.ones(2,2,4,4)

    SGO = Operator_Layer(1,1, ks=1)
    print("Weights")
    print(SGO.operator.weight)
    print("Bias")
    print(SGO.bias)

    print(SGO(ip))

