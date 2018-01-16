"""DatasetTest.py

This module gives unit testing classes for VAE Encoder/Decoder structures.
"""

import unittest
import sys
# Add the src folder path to the sys.path list
sys.path.append('../src')

from EncoderDecoder import Encoder
from EncoderDecoder import Decoder


#------------------------ Begin class TestEncoder ----------------------------

class TestEncoder(unittest.TestCase):
    """Tests encoder structure"""

    #---------------------------------------

    def test_wrong_inputDim(self):
        inputDim = 512
        outputDim = 6
        dimValues = [513, 128, 6]
        e = Encoder(inputDim, dimValues, outputDim)
        self.assertFalse(e.created)

    #---------------------------------------

    def test_wrong_outputDim(self):
        inputDim = 513
        outputDim = 7
        dimValues = [513, 128, 6]
        e = Encoder(inputDim, dimValues, outputDim)
        self.assertFalse(e.created)

    #---------------------------------------

    def test_good_IODim(self):
        inputDim = 513
        outputDim = 6
        dimValues = [513, 128, 6]
        e = Encoder(inputDim, dimValues, outputDim)
        self.assertTrue(e.created)

    #---------------------------------------

    def test_multiLayerNN(self):
        inputDim = 513
        outputDim = 6
        dimValues = [513, 128, 256, 64, 6]
        e = Encoder(inputDim, dimValues, outputDim)
        self.assertTrue(e.nb_h == 3)

    #---------------------------------------

    def test_wrong_NN(self):
	    inputDim = 513
	    outputDim = 6
	    dimValues = [513, 6]
	    e = Encoder(inputDim, dimValues, outputDim)
	    self.assertFalse(e.created)

#------------------------ End class TestEncoder ------------------------------

#------------------------ Begin class TestDecoder ----------------------------

class TestDecoder(unittest.TestCase):
    """Tests decoder structure"""

    #---------------------------------------

    def test_wrong_inputDim(self):
        inputDim = 6
        outputDim = 512
        dimValues = [6, 128, 513]
        d = Decoder(inputDim, dimValues, outputDim)
        self.assertFalse(d.created)

    #---------------------------------------

    def test_wrong_outputDim(self):
        inputDim = 7
        outputDim = 513
        dimValues = [6, 128, 513]
        d = Decoder(inputDim, dimValues, outputDim)
        self.assertFalse(d.created)

    #---------------------------------------

    def test_good_IODim(self):
        inputDim = 6
        outputDim = 513
        dimValues = [6, 128, 513]
        d = Decoder(inputDim, dimValues, outputDim)
        self.assertTrue(d.created)

    #---------------------------------------

    def test_multiLayerNN(self):
        inputDim = 6
        outputDim = 513
        dimValues = [6, 64, 256, 128, 513]
        d = Decoder(inputDim, dimValues, outputDim)
        self.assertTrue(d.nb_h == 4)

    #---------------------------------------

    def test_wrong_NN(self):
	    inputDim = 6
	    outputDim = 513
	    dimValues = [6]
	    d = Decoder(inputDim, dimValues, outputDim)
	    self.assertFalse(d.created)

    #---------------------------------------

    # def test_gaussianDecoder(self):

    #---------------------------------------

    # def test_wrong_gaussianDecoder(self):

    #---------------------------------------

#------------------------ End class TestDecoder ------------------------------

#---------------------------- Test suites ------------------------------------

suiteEncoder = unittest.TestLoader().loadTestsFromTestCase(TestEncoder)
print "\n\n------------------- Encoder Test Suite -------------------\n"
unittest.TextTestRunner(verbosity=2).run(suiteEncoder)
suiteDecoder = unittest.TestLoader().loadTestsFromTestCase(TestDecoder)
print "\n\n------------------- Decoder Test Suite -------------------\n"
unittest.TextTestRunner(verbosity=2).run(suiteDecoder)
