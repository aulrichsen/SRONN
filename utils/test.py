
from email.errors import NonPrintableDefect
import torch
import torch.nn as nn

"""
Functions and classes to help run unittests
"""

class Test_Parser:
    """
    Get training arguments to use in function without using argparse.
    """
    def __init__(self):
        self.model="SRONN"
        self.q=3
        self.dataset="PaviaU"
        self.scale=2
        self.epochs=10
        self.lr=0.0001
        self.lr_milestones=[2500, 8000]
        self.optimizer="Adam"
        self.loss_function="MSE"
        self.metrics_step=10
        self.grad_clip=False
        self.clip_val=1
        self.weight_transfer=False
        self.checkpointt=""
        self.SR_kernel=False
        self.bs=64
        self.SISR=False
        self.wandb_group=None
        self.wandb_jt=None


class Test_Model_Identical(nn.Module):
    """
    Fake model to produce identiacal output for testing
    """
    def __init__(self):
        super(Test_Model_Identical, self).__init__()
        
        self.name = "Test_Model_Identical"
        self.num_params = 7357  # Supposed to look like TEST
        self.conv = nn.Conv2d(10,10,3)

    def forward(self, x):
        return x    # Output identical to input

class Test_Model_Different(nn.Module):
    """
    Fake model to produce different output for testing
    """
    def __init__(self, channels):
        super(Test_Model_Different, self).__init__()
        
        self.name = "Test_Model_Different"
        self.num_params = 7357  # Supposed to look like TEST
        self.conv = nn.Conv2d(channels,channels,3, padding='same')

    def forward(self, x):
        return self.conv(x)  
