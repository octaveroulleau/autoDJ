"""EncoderDecoder.py

This module gives classes for VAE's encoder and decoder structures.
It is used in ./VAE.py
Unit testing: see ../unitTest/EncoderDecoderTest.py

Encoder and Decoder classes are almost the same except that the Decoder
class can handle bernoulli/gaussian types.

*If the Decoder is gaussian type, then Encoder and Decoder have the same
structure. The decoder output space is modeled by a gaussian
distribution like it is for the latent space. So both encoder and
decoder ultimate layers are dedicated to compute multidimensional mean
and variance (dimZ for latent space, dimX for output space). Prior
layers are hidden layers with non linear activation function.
VAE output considered is here X_mu, while both X_mu and X_logSigma are
used to compute loss and try to maximize lower bound.
Example for dimX=1024, dimZ=10, dimO_hLayer=600:
    encoder:
    [inputData:1024] -> {hLayer:1024-600} -> {NL function:"relu"} --> {"mean z" layer}-----------> [z_mu:10]
                                                                  |
                                                                  --> {"log(variance z)" layer} -> [z_logSigma:10]
    decoder:
    [inputData:10] -> {hLayer:10-600} -> {NL function:"relu"} --> {"mean X_estimated" layer} ----------> [X_mu:1024]
                                                              |
                                                              --> {"log(variance X_estimated)" layer} -> [X_logSigma:1024]

*Else if it is bernoulli type, then Decoder has another hidden layer
with sigmoid NL function. It is to clip values between 0 and 1.
Hence this type decoder is adapted for binary outputs.
VAE output considered is here X_sampled and used to compute loss and
try to maximize lower bound.
Example for dimX=1024, dimZ=10, dimO_hLayer=600:
    encoder:
    Same as before

    decoder:
    [inputData:10] -> {hLayer:10-600} -> {NL function:"relu"} -> {hLayer:600-1024} -...
    ...-> {NL function:"sigmoid"} -> [X_sampled:1024]

NB: for either bernoulli or gaussian type, see Decoder class in ./EncoderDecoder.py

Todo:
    * merge common parts of Encoder/Decoder
    * error handler

"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


#---------------------------- Begin class Encoder ------------------------

class Encoder(nn.Module):
    """A nn.Module which gives the VAE a structure to encode input data to latent space.

    Attributes:

        dimX (int):             input dimension (data space)
        dimZ (int):             output dimension (latent space)

        nb_h (int):             number of hidden layers
        inDim_h (list):         Input dimensions of each layer (including z_mu and
                                z_logSigma layers) (e.g. [1024, 600])
        outDim_h (list):        Output dimensions of each layer (including z_mu and
                                z_logSigma layers) (e.g. [600, 10])
        weights_h (list):       list of weights values for each hidden layer (excluding
                                z_mu and z_logSigma layers))
        bias_h (list):          list of bias values for each hidden layer (excluding
                                z_mu and z_logSigma layers))

        weight_mu (list):       weights values for z_mu layer
        bias_mu (list):         bias values for z_mu layer
        weight_logSigma (list): weights values for z_logSigma layer
        bias_logSigma (list):   bias values for z_logSigma layer

        created (bool):         flags on encoder current state
    """

    def __init__(self, inputDim, dimValues, outputDim):
        """Create an encoder strucutre by setting all weights and bias layers needed

        Args:
            inputDim (int):     input dimension (data space)
            outputDimZ (int):   output dimension (latent space)
            dimValues (list):   dimension values of whole encoder (e.g. [1024, 600, 10])
        """

        # superclass init
        super(Encoder, self).__init__()
        self.created = False

        # dimension of inputs X
        self.dimX = inputDim

        # dimension of outputs Z
        self.dimZ = outputDim

        # Encoder NN structure:
        # define HIDDEN layers number
        self.nb_h = len(dimValues) - 2
        # check if args match
        if self.nb_h < 1:
            print ("ERROR_Encoder: Not enough dimension values")
            return None
        elif self.dimX != dimValues[0]:
            print ("ERROR_Encoder: X & NN input dimensions mismatched")
            return None
        elif self.dimZ != dimValues[len(dimValues) - 1]:
            print ("ERROR_Encoder: Z & NN output dimensions mismatched")
            return None

        # store IO dimensions for each layers in a list
        #& create initialized hidden layers of the NN
        self.inDim_h = []
        self.outDim_h = []

        self.weights_h = []
        self.bias_h = []
        self.weight_mu = None
        self.bias_mu = None
        self.weight_logSigma = None
        self.bias_logSigma = None

        for index_h in range(self.nb_h):
            self.inDim_h.append(dimValues[index_h])
            self.outDim_h.append(dimValues[index_h + 1])
            self.weights_h.append(
                xavier_init(size=[self.inDim_h[index_h], self.outDim_h[index_h]]))
            self.bias_h.append(
                Variable(torch.zeros(self.outDim_h[index_h]), requires_grad=True))

        # LAST LAYER is made by hand whereas for bernoulli DECODER IT'S NOT
        self.weight_mu = xavier_init(
            size=[self.outDim_h[self.nb_h - 1], self.dimZ])
        self.bias_mu = Variable(torch.zeros(self.dimZ), requires_grad=True)
        self.weight_logSigma = xavier_init(
            size=[self.outDim_h[self.nb_h - 1], self.dimZ])
        self.bias_logSigma = Variable(
            torch.zeros(self.dimZ), requires_grad=True)

        self.created = True

    #---------------------------------------

    def getInfo(self):
        print('\nEncoder net : ')
        for idx in range(self.nb_h):
            print('layer ' + str(idx) + ': size ' +
                  str(self.weights_h[idx].size(0)))

#---------------------------- End class Encoder --------------------------

#---------------------------- Begin class Decoder ------------------------


class Decoder(nn.Module):
    """A nn.Module which gives the VAE a structure to decode data from latent space.
    Type is either bernoulli or gaussian (see below)

    Attributes:

        dimZ (int):             input dimension (latent space)
        dimX (int):             output dimension (data space)

        nb_h (int):             number of hidden layers
        inDim_h (list):         Input dimensions of each layer (if gaussian, including
                                z_mu and z_logSigma layers) (e.g. [1024, 600])
        outDim_h (list):        Output dimensions of each layer (if gaussian, including
                                z_mu and z_logSigma layers) (e.g. [600, 10])
        weights_h (list):       list of weights values for each hidden layer (if gaussian,
                                excluding z_mu and z_logSigma layers))
        bias_h (list):          list of bias values for each hidden layer (if gaussian,
                                excluding z_mu and z_logSigma layers) i)

        bernoulli (bool):       flag on bernoulli decoder type
        gaussian (bool):        flag on gaussian decoder type

        Attributes existing only when gaussian decoder type:
        weight_mu (list):       weights values for z_mu layer
        bias_mu (list):         bias values for z_mu layer
        weight_logSigma (list): weights values for z_logSigma layer
        bias_logSigma (list):   bias values for z_logSigma layer



        created (bool):         flags on decoder current state
    """

    def __init__(self, inputDim, dimValues, outputDim, bernoulli=True, gaussian=False):
        """Create an decoder strucutre by setting all weights and bias layers needed
        for either bernoulli or gaussian decoder type

        Args:
            inputDim (int):     input dimension (data space)
            outputDimZ (int):   output dimension (latent space)
            dimValues (list):   dimension values of whole decoder (e.g. [10, 600, 1024])
            bernoulli (bool):   flag on bernoulli decoder type
            gaussian (bool):    flag on gaussian decoder type
        """

        # superclass init
        super(Decoder, self).__init__()
        self.created = False

        # dimension of inputs Z
        self.dimZ = inputDim

        # dimension of outputs X
        self.dimX = outputDim

        # decoder type flags
        self.bernoulli = bernoulli
        self.gaussian = gaussian

        # Decoder NN structure:
        # define HIDDEN layers number
        if self.bernoulli and not self.gaussian:
            self.nb_h = len(dimValues) - 1
        elif self.gaussian and not self.bernoulli:
            self.nb_h = len(dimValues) - 2
        else:
            print("ERROR_Decoder: Decoder type unknown")
            raise
        # check if args match
        if self.nb_h < 1:
            print ("ERROR_Decoder: Not enough dimension values")
            return None
        elif self.dimZ != dimValues[0]:
            print ("ERROR_Decoder: Z & NN input dimensions mismatched")
            return None
        elif self.dimX != dimValues[len(dimValues) - 1]:
            print ("ERROR_Decoder: X & NN output dimensions mismatched")
            return None

        # store IO dimensions for each layers in a list
        #& create hidden layers of the NN (private)
        self.inDim_h = []
        self.outDim_h = []

        self.weights_h = []
        self.bias_h = []

        if gaussian and not bernoulli:
            self.weight_mu = None
            self.bias_mu = None
            self.weight_logSigma = None
            self.bias_logSigma = None

        for index_h in range(self.nb_h):
            self.inDim_h.append(dimValues[index_h])
            self.outDim_h.append(dimValues[index_h + 1])
            self.weights_h.append(
                xavier_init(size=[self.inDim_h[index_h], self.outDim_h[index_h]]))
            self.bias_h.append(
                Variable(torch.zeros(self.outDim_h[index_h]), requires_grad=True))

        if gaussian and not bernoulli:
            # LAST LAYER is made by hand whereas for gaussian decoder
            self.weight_mu = xavier_init(
                size=[self.outDim_h[self.nb_h - 1], self.dimX])
            self.bias_mu = Variable(torch.zeros(self.dimX), requires_grad=True)
            self.weight_logSigma = xavier_init(
                size=[self.outDim_h[self.nb_h - 1], self.dimX])
            self.bias_logSigma = Variable(
                torch.zeros(self.dimX), requires_grad=True)

        self.created = True

    #---------------------------------------

    def getInfo(self):
        print('\nDecoder net : ')
        for idx in range(self.nb_h):
            print('layer ' + str(idx) + ': size ' +
                  str(self.weights_h[idx].size(0)))

#---------------------------- End class Encoder --------------------------

#---------------------------- Begin functions ---------------------------------


def xavier_init(size):
    """Xavier init to initialize Variable in Encoder/Decoder's nets"""
    in_dim = size[0]
    xavier_stddev = 1 / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)
