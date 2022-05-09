import unittest

import numpy as np

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

class Test_Metrics(unittest.TestCase):
    """
    Test evaluation metrics
    """

    def test_psnr(self):
        """
        test dimenstion of data produced by bicubic_lr function 
        test data produced is not identical to input
        """

        c = 103

        test_ip = np.random.rand(64,64,c)     # x, y, c
        test_tar = np.random.rand(64,64,c)

        all_psnr = psnr(test_ip, test_tar)      # psnr of all channels at once

        ind_chan_psnr = [psnr(test_ip[:,:,i], test_tar[:,:,i]) for i in range(c)]

        self.assertAlmostEqual(all_psnr, np.mean(ind_chan_psnr), msg="All PSNR not equal to mean individual channel PSNR.", delta=0.001)

    def test_ssim(self):
        c = 103

        test_ip = np.random.rand(64,64,c)     # x, y, c
        test_tar = np.random.rand(64,64,c)

        all_ssim = ssim(test_ip, test_tar)      # psnr of all channels at once

        ind_chan_ssim = [ssim(test_ip[:,:,i], test_tar[:,:,i]) for i in range(c)]

        self.assertAlmostEqual(all_ssim, np.mean(ind_chan_ssim), msg="All SSIM not equal to mean individual channel SSIM.", delta=0.001)
