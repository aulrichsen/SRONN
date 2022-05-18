import unittest

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from Training import train, eval
from Load_Data import get_dataloaders, get_data
from utils.test import Test_Model_Identical, Test_Model_Different, Test_Parser


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
                self.dataset="PaviaU"
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


    def test_train(self):

        opt = Test_Parser()
        opt.epochs = 3
        opt.model = "Test_Model_Different"      
        opt.SISR = False 

        train_dl, val_dl, test_dl, channels, dataset_name = get_dataloaders(opt, "cpu")

        model = Test_Model_Different(channels)

        psnrs, ssims, sams = train(model, train_dl, val_dl, test_dl, opt, jt=dataset_name)

        self.assertEqual(len(psnrs), opt.epochs, msg="train returned incorrect number of PSNR values.")
        self.assertEqual(len(ssims), opt.epochs, msg="train returned incorrect number of SSIM values.")
        self.assertEqual(len(sams), opt.epochs, msg="train returned incorrect number of SAM values.")