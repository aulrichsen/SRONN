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

        #self.operator = nn.Conv2d(in_channels*q_order, out_channels, ks,padding='same')
        self.conv1 = nn.Conv2d(in_channels*q_order, 64, kernel_size=9, padding='same')
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding='same')
        self.conv3 = nn.Conv2d(32, out_channels, kernel_size=5, padding='same')
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Raise inputs up to power q 
        power_input = [x]
        for q in range(2, self.q_order+1):
            power_input.append(torch.pow(x, q))
        power_input = torch.cat(power_input, 1)

        #operator_out = self.operator(power_input) #torch.relu(self.operator(power_input),inplace=False)
        x = self.relu(self.conv1(power_input))
        x = self.relu(self.conv2(x))
        operator_out = self.conv3(x)
        return operator_out

if __name__ == "__main__":
    ip = torch.ones(2,2,4,4)

    SGO = Operator_Layer(1,1, ks=1)
    print("Weights")
    print(SGO.operator.weight)
    print("Bias")
    print(SGO.bias)

    print(SGO(ip))

