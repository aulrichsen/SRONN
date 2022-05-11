import unittest

#from unit_tests.test_Load_Data import Test_Load_Data
from unit_tests.test_Training import Test_Training
#from unit_tests.test_metrics import Test_Metrics

"""
This file necessary due to test file hierarchy, relative imports cannot be done when running code directly from individual test scrips contained within tests folder. i.e. when __name__ == "__main__"
The relative imports are necessary in order to import functions and methods from partent folder to be tested. Hence why this script is contained in the main folder.
"""

if __name__ == '__main__':
    unittest.main()