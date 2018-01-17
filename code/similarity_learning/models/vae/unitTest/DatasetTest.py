"""DatasetTest.py

This module gives unit testing classes for handling .npz datasets
"""

import unittest
import sys
# Add the src folder path to the sys.path list
# sys.path.append('../src/dataset')
sys.path.append('../src/')

from ManageDataset import NPZ_Dataset


#---------------------------- Begin class TestNPZDataset ----------------

class TestNPZDataset(unittest.TestCase):
    """Tests .npz datasets"""

    #---------------------------------------

    def testDatasetLoad(self):
        """check loading dataset"""
        dataset = NPZ_Dataset('dummyDataset98.npz',
                              './dummyDataset/', 'Spectrums', 'labels')
        for i in range(5):
            data = dataset[i]
            print(i, data['image'], data['label'])
        self.assertTrue(dataset != [])

    #---------------------------------------

    def testDatasetLength(self):
        """check the length of the dataset"""
        dataset = NPZ_Dataset('dummyDataset98.npz',
                              './dummyDataset/', 'Spectrums', 'labels')
        self.assertTrue(len(dataset) == 98)

#---------------------------- End class TestNPZDataset -------------------

#---------------------------- Test suites ------------------------------------

suiteNPZDataset = unittest.TestLoader().loadTestsFromTestCase(TestNPZDataset)
print("\n\n------------------- Dataset from .npz file Test Suite -------------------\n")
unittest.TextTestRunner(verbosity=2).run(suiteNPZDataset)

