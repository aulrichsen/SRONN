import unittest

import numpy as np

from Load_Data import get_pavia_data, bicubic_lr

class Test_Load_Data(unittest.TestCase):

    def test_bicubic_lr(self):
        test_img = np.random.rand(103,64,64)

        test_out = bicubic_lr(test_img, 2)

        self.assertListEqual(list(test_img.shape), list(test_out.shape), msg="Output of bicubic_lr not the same dimensions as input.")

        mse = ((test_img - test_out)**2).mean(axis=None)

        self.assertGreater(mse, 0, "bicubic_lr input and output identical")