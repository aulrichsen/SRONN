import unittest

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from Training import eval
from Load_Data import get_dataloaders, get_data


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
        test_data = TensorDataset(test_img, test_img)
        test_dl = DataLoader(test_data, batch_size=64, shuffle=False)

        test_model = Test_Model_Identical().to(self.device)

        psnr, ssim, sam = eval(test_model, test_dl)

        self.assertGreater(psnr, 100, msg="PSNR not high enough for identical.")
        self.assertEqual(ssim, 1, msg='SSIM not 1 for identical input and target.')
        self.assertEqual(sam, 0, msg='SAM not 0 for identical input and output.')

    def test_eval_different(self):
        test_img = torch.rand(41,103,64,64).to(self.device)
        test_tar = torch.rand(41,103,64,64)
        test_data = TensorDataset(test_img, test_tar)
        test_dl = DataLoader(test_data, batch_size=64, shuffle=False)

        test_model = Test_Model_Different(channels=103).to(self.device)

        psnr, ssim, sam = eval(test_model, test_dl)

        self.assertLess(psnr, 100, msg="PSNR too high for different input output.")
        self.assertLess(ssim, 1, msg='SSIM too high for different input and target.')
        self.assertGreater(sam, 0, msg='SAM too low for different input and output.')

    def test_eval_SISR(self):

        class Parser():
            def __init__(self):
                self.dataset="Pavia"
                self.scale=2
                self.SR_kernel=False
                self.SISR=True
                self.bs=64

        test_opt = Parser()

        train_dl, val_dl, test_dl, channels, dataset_name = get_dataloaders(test_opt, "cpu")
        
        test_model = Test_Model_Different(channels=1)

        psnr, ssim, sam = eval(test_model, test_dl, SISR=True)

        self.assertLess(psnr, 100, msg="PSNR too high for different input output.")
        self.assertLess(ssim, 1, msg='SSIM too high for different input and target.')
        self.assertGreater(sam, 0, msg='SAM too low for different input and output.')


    def test_SISR_eval(self):
        x_train, y_train, _, _, _, _, _ = get_data(dataset="PaviaU")

        x_test = x_train[[0,10]]
        y_test = y_train[[0,10]]

        # Regular
        test_dl = DataLoader(TensorDataset(x_test, y_test), batch_size=1)
        test_model = Test_Model_Identical()
        psnr, ssim, sam = eval(test_model, test_dl)

        # SISR equivalent data
        tile_idxs = torch.cat((torch.zeros(x_train.shape[1]), torch.ones(x_train.shape[1])), dim=0)
        SISR_x_test = x_test.reshape(-1, 1, x_test.shape[-2], x_test.shape[-1])
        SISR_y_test = y_test.reshape(-1, 1, y_test.shape[-2], y_test.shape[-1])
        
        mse = nn.MSELoss()
        with self.subTest():
            # Test SISR data shaped correctly'
            self.assertEqual(mse(x_test[0], SISR_x_test[:x_test.shape[1]].squeeze()), 0, msg="SISR x data not shaped correctly.")
            self.assertEqual(mse(y_test[0], SISR_y_test[:x_test.shape[1]].squeeze()), 0, msg="SISR y data not shaped correctly.")
        
        test_dl = DataLoader(TensorDataset(SISR_x_test, SISR_y_test), batch_size=1)
        test_dl.tile_idxs = tile_idxs

        SISR_psnr, SISR_ssim, SISR_sam = eval(test_model, test_dl, SISR=True)

        with self.subTest():
            # Test PSNR the same between regular and SISR
            self.assertEqual(psnr, SISR_psnr, msg="SISR PSNR not correct.")

        with self.subTest():
            # Test SSIM the same between regular and SISR
            self.assertEqual(ssim, SISR_ssim, msg="SISR SSIM not correct.")

        with self.subTest():
            # Test SAM the same between regular and SISR
            self.assertEqual(sam, SISR_sam, msg="SISR SAM not correct.")


    def test_PSNR(self):
        # Test sklearn psnr function

        test_chans = 5
        ip1 = np.random.rand(64, 64, test_chans)
        tar1 = np.random.rand(64, 64, test_chans)
        all_psnr1 = psnr(ip1, tar1, data_range=1)
        indi_psnrs1 = [psnr(ip1[:,:,i], tar1[:,:,i], data_range=1) for i in range(test_chans)]
        self.assertAlmostEqual(all_psnr1, np.mean(indi_psnrs1), msg="Full PSNR does not match individual slice PSNR average for single tile.", delta=0.001)

        """
        # ** Will cause failure if psnr of full HSI images averaged vs individual slices. **

        ip2 = np.random.rand(64, 64, test_chans)
        tar2 = np.random.rand(64, 64, test_chans)
        all_psnr2 = psnr(ip2, tar2, data_range=1)
        indi_psnrs2 = [psnr(ip2[:,:,i], tar2[:,:,i], data_range=1) for i in range(test_chans)]
        
        ip3 = np.random.rand(64, 64, test_chans)
        tar3 = np.random.rand(64, 64, test_chans)
        all_psnr3 = psnr(ip3, tar3, data_range=1)
        indi_psnrs3 = [psnr(ip3[:,:,i], tar3[:,:,i], data_range=1) for i in range(test_chans)]
        
        self.assertAlmostEqual(np.mean([all_psnr1, all_psnr2, all_psnr3]), np.mean(indi_psnrs1+indi_psnrs2+indi_psnrs3), msg="Full PSNR does not match individual slice PSNR average for three tiles.", delta=0.001)
        """