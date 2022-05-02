import unittest

import numpy as np
import torch
import torch.nn as nn

from Load_Data import get_all_data, get_data, bicubic_lr

class Test_Load_Data(unittest.TestCase):

    def test_bicubic_lr(self):
        """
        test dimenstion of data produced by bicubic_lr function 
        test data produced is not identical to input
        """

        test_img = np.random.rand(103,64,64)

        test_out = bicubic_lr(test_img, 2)

        self.assertListEqual(list(test_img.shape), list(test_out.shape), msg="Output of bicubic_lr not the same dimensions as input.")

        mse = ((test_img - test_out)**2).mean(axis=None)

        self.assertGreater(mse, 0, "bicubic_lr input and output identical")
    
    def test_data_range(self):

        datasets = ["Botswana", 'Cuprite', 'Indian_Pines', "KSC", "Pavia", "Salinas", "Urban", "PaviaU"]

        for ds in datasets:
            x_train, y_train, x_val, y_val, x_test, y_test, _ = get_data(dataset=ds)
            with self.subTest():
                #self.assertEqual(max(torch.max(x_train), torch.max(x_val), torch.max(x_test)), 1.0, msg=f"{ds} X max not 1")
                #self.assertEqual(max(torch.min(x_train), torch.min(x_val), torch.min(x_test)), 0.0, msg=f"{ds} X min not 1")
                self.assertEqual(max(torch.max(y_train), torch.max(y_val), torch.max(y_test)), 1.0, msg=f"{ds} Y max not 1")
                self.assertEqual(max(torch.min(y_train), torch.min(y_val), torch.min(y_test)), 0.0, msg=f"{ds} Y min not 1")

    def test_reproducible(self):
        """
        Test if data produced by get_pavia_data function is reproducible
        """

        x1_train, y1_train, x1_val, y1_val, x1_test, y1_test, _ = get_data()

        x2_train, y2_train, x2_val, y2_val, x2_test, y2_test, _ = get_data()

        mse_loss = nn.MSELoss()     # MSE function to compare sperate instances of data are identical

        self.assertEqual(mse_loss(x1_train, x2_train), 0, msg="x_train not reproducible.")
        self.assertEqual(mse_loss(y1_train, y2_train), 0, msg="y_train not reproducible.")
        self.assertEqual(mse_loss(x1_val, x2_val), 0, msg="x_val not reproducible.")
        self.assertEqual(mse_loss(y1_val, y2_val), 0, msg="x_train not reproducible.")
        self.assertEqual(mse_loss(x1_test, x2_test), 0, msg="x_train not reproducible.")
        self.assertEqual(mse_loss(y1_test, y2_test), 0, msg="x_train not reproducible.")


    def test_band_removal(self):
        """
        test that band removal from dataset works correctly
        """

        tolerance = 0.0001  # mse tolerance value, since band removal will affect normalisation

        bands_to_remove = [0,1,2,-3,-2,-1]

        x1_train, y1_train, x1_val, y1_val, x1_test, y1_test, _ = get_data(bands_to_remove=bands_to_remove)

        x2_train, y2_train, x2_val, y2_val, x2_test, y2_test, _ = get_data()

        mse_loss = nn.MSELoss()     # MSE function to compare sperate instances of data are identical

        self.assertLess(mse_loss(x1_train, x2_train[:, 3:-3, :, :]), tolerance, msg="x_train bands not removed correctly.")
        self.assertLess(mse_loss(y1_train, y2_train[:, 3:-3, :, :]), tolerance, msg="y_train bands not removed correctly.")
        self.assertLess(mse_loss(x1_val, x2_val[:, 3:-3, :, :]), tolerance, msg="x_val bands not removed correctly.")
        self.assertLess(mse_loss(y1_val, y2_val[:, 3:-3, :, :]), tolerance, msg="x_train bands not removed correctly.")
        self.assertLess(mse_loss(x1_test, x2_test[:, 3:-3, :, :]), tolerance, msg="x_train bands not removed correctly.")
        self.assertLess(mse_loss(y1_test, y2_test[:, 3:-3, :, :]), tolerance, msg="x_train bands not removed correctly.")

    """
    def test_get_all_data(self):
        train_data, val_data, test_data = get_all_data()

        for i in range(10):
            X = val_data.X[i+1+(i+1)*10, i*10].cpu()
            Y = val_data.Y[i+1+(i+1)*10, i*10].cpu()
            with self.subTest():
                self.assertEqual(torch.max(X), 1.0, msg=f"all val X {i} {i*10} max not 1")
                self.assertEqual(torch.min(X), 0.0, msg=f"all val X {i} {i*10} min not 0")
                self.assertEqual(torch.max(Y), 1.0, msg=f"all val Y {i} {i*10} max not 1")
                self.assertEqual(torch.min(Y), 0.0, msg=f"all val Y {i} {i*10} min not 0")
    """   
