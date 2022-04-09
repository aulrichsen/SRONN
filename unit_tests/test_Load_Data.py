import unittest

import numpy as np
import torch.nn as nn

from Load_Data import get_data, bicubic_lr

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

