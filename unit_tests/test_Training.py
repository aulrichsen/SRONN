import unittest

import numpy as np
import torch
import torch.nn as nn

from Training import eval

class Test_Model_Identical(nn.Module):
    def __init__(self):
        super(Test_Model_Identical, self).__init__()
        
        self.conv = nn.Conv2d(10,10,3)

    def forward(self, x):
        return x    # Output identical to input

class Test_Model_Different(nn.Module):
    def __init__(self, channels):
        super(Test_Model_Different, self).__init__()
        
        self.conv = nn.Conv2d(channels,channels,3, padding='same')

    def forward(self, x):
        return self.conv(x)    # Output identical to input


class Test_Training(unittest.TestCase):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_eval_identical(self):
        test_img = torch.rand(41,103,64,64)

        test_model = Test_Model_Identical().to(self.device)

        psnr, ssim, sam = eval(test_model, test_img, test_img)

        self.assertGreater(psnr, 100, msg="PSNR not high enough for identical.")
        self.assertEqual(ssim, 1, msg='SSIM not 1 for identical input and target.')
        self.assertEqual(sam, 0, msg='SAM not 0 for identical input and output.')

    def test_eval_different(self):
        test_img = torch.rand(41,103,64,64).to(self.device)
        test_tar = torch.rand(41,103,64,64)

        test_model = Test_Model_Different(channels=103).to(self.device)

        psnr, ssim, sam = eval(test_model, test_img, test_tar)

        self.assertLess(psnr, 100, msg="PSNR too high for different input output.")
        self.assertLess(ssim, 1, msg='SSIM too high for different input and target.')
        self.assertGreater(sam, 0, msg='SAM too low for different input and output.')

    