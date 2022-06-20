import os
import glob
import unittest

import torch
import torch.nn as nn

from Test_Model import test_model
from Load_Data import get_dataloaders
from utils.test import Test_Model_Different, Test_Parser

class Test_Test_Model(unittest.TestCase):
    """
    Test Test_Model file functions
    """

    def test_test_model(self):
        """
        Test test_model function.
        """

        disp_slices = [{"b": 0, "c": 0}, {"b": 10, "c": 0}]

        save_dir = "test_model_test"

        opt = Test_Parser()

        _, _, test_dl, channels, _ = get_dataloaders(opt, "cpu")

        model = Test_Model_Different(channels=channels)

        test_model(model, test_dl, opt, save_dir=save_dir, disp_slices=disp_slices)

        # Check Relevalnt files have been made
        self.assertTrue(os.path.exists("Results/"+save_dir+"/objective/info.txt"), msg="Info file not created.")
        num_objective_files = len(glob.glob("Results/"+save_dir+"/objective/*"))
        self.assertEqual(num_objective_files, 3, msg="Incorrect number of files created in objective folder.")

        self.assertTrue(os.path.exists("Results/"+save_dir+"/true"), msg="True directory not created.")
        num_true_files = len(glob.glob("Results/"+save_dir+"/true/*"))
        self.assertEqual(num_true_files, 2, msg="Incorrect number of files created in true folder.")


        # Repeat for SISR
        opt.SISR=True
        save_dir = "test_model_test_SISR"
        _, _, test_dl, channels, _ = get_dataloaders(opt, "cpu")

        model = Test_Model_Different(channels=channels)

        test_model(model, test_dl, opt, save_dir=save_dir, disp_slices=disp_slices)

        # Check Relevalnt files have been made
        self.assertTrue(os.path.exists("Results/"+save_dir+"/objective/info.txt"), msg="SISR Info file not created.")
        num_objective_files = len(glob.glob("Results/"+save_dir+"/objective/*"))
        self.assertEqual(num_objective_files, 3, msg="Incorrect number of files created in SISR objective folder.")

        self.assertTrue(os.path.exists("Results/"+save_dir+"/true"), msg="SISR True directory not created.")
        num_true_files = len(glob.glob("Results/"+save_dir+"/true/*"))
        self.assertEqual(num_true_files, 2, msg="Incorrect number of files created in SISR true folder.")
