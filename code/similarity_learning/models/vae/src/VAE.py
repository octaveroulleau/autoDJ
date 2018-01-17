"""VAE.py

This module gives a class and functions for VAE creation training and generation.
It is used in ./mainScript.py
Unit testing: see ../unitTest/VAETest.py

Plus the vanilla VAE, it has
    -modular encoder and decoder: can change quickly number/dim of layers and NL functions
    -a warm-up option
    -a bernoulli type: reconstruction loss is computed from X_sample using binary cross binary_cross_entropy
    -a gaussian type: recon loss is computed from log(likelihood) of a gaussian distribution (X_mu, X_logSigma)
    -a save/load workflow

NB: for either bernoulli or gaussian type, see Decoder class in ./EncoderDecoder.py 

Todo:
    * test reconstruction
    * more generation functions
    * batch normalization
    * error handler

.. Adapted from:
   https://github.com/wiseodd/generative-models
   https://github.com/pytorch/examples/tree/master/vae
"""

import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.autograd import Variable
from torchvision.utils import save_image
import torch.utils.data
from torchvision import datasets, transforms
sys.path.append('similarity_learning/models/vae/src')
from EncoderDecoder import Encoder, Decoder

#---------------------------- Begin class VAE ---------------------------------

class VAE(nn.Module):
    """A nn.Module which enables VAE trainings on datasets and loadings, generatings.

    Attributes:

        encoder (Encoder):      encoder structure
        decoder (Decoder):      decoder structure
        IOh_dims_Enc (list):    IO dimensions of encoder's layers (e.g. [1024, 600, 10])
        IOh_dims_Dec (list):    IO dimensions of decoder's layers (e.g. [10, 600, 1024])
        NL_funcE (list):        layers' non linear functions of encoder (e.g. ['relu'])
        NL_funcD (list):        layers' non linear functions of decoder (e.g. ['relu', 'sigmoid'])

        parameters (list):      gather weights and bias of each encoder/decoder layers
        z_mu (Variable):        latent gaussian distributions means
        z_logSigma (Variable):  latent gaussian distributions variances
        X_sample (Variable):    VAE output, same dim than VAE input (used in bernoulli VAE)
        X_mu (Variable):        VAE output, same dim than VAE input (used in gaussian VAE)
        X_logSigma (Variable):  infers output distributions variances (used in gaussian VAE)

        mb_size (int):          minibatch size
        lr (float):             learning rate
        beta (float):           coefficient for regularization term in total loss
        Nwu (int):              warm-up time in epochs number (default 1 -> no warm-up)
        beta_wu (int):          beta current value (linear increase across epochs)
        beta_inc (float):       beta increment during warm-up
        epoch_nb (int):         training epochs number               
        recon_loss (list):      reconstruction loss recorder
        regul_loss (list):      regularization loss recorder

        noise_in (bool):        flag on noising input during training
        nois_gain (float):      noise gain if noising input

        created/trained/saved/loaded (bool):
                                flags on VAE current state
    """

    #---------------------------------------

    def __init__(self, X_dim, Z_dim, IOh_dims_Enc, IOh_dims_Dec, NL_types_Enc, NL_types_Dec,
                 mb_size=64, beta=1, Nwu=1, lr=1e-3, bernoulli=True, gaussian=False, noiseIn=False, noiseGain=0.):
        """Create a VAE strucutre by setting all attributes and calling Encoder/Decoder constructors.

        Args:
            X_dim (int):            input (data) and output dimension (e.g. 1024)
            Z_dim (int):            latent space dimension  (e.g. 10)
            IOh_dims_Enc (list):    IO dimensions of encoder's layers (e.g. [1024, 600, 10])
            IOh_dims_Dec (list):    IO dimensions of decoder's layers (e.g. [10, 600, 1024])
            NL_types_Enc (list):    layers' non linear functions of encoder (e.g. ['relu'])
            NL_types_Dec (list):    layers' non linear functions of decoder (e.g. ['relu', 'sigmoid'])
            mb_size (int):          minibatch size (default 64)
            lr (float):             learning rate (default 0.001)
            beta (float):           coefficient for regularization term (e.g kll) in total loss (default 3)
            Nwu (int):              warm-up time in epochs number (default 50)
            bernoulli (bool):       flag for bernoulli VAE type (default True)
            gaussian (bool):        flag for gaussian VAE type (default False)

            noiseIn (bool):         noise input decoder data when training (default False)
            noiseGain (float)       noise gain if noiseIn is True (default 0.)
        """

        # superclass init
        super(VAE, self).__init__()
        self.created = False

        self.IOh_dims_Enc = IOh_dims_Enc
        self.IOh_dims_Dec = IOh_dims_Dec

        self.encoder = Encoder(X_dim, self.IOh_dims_Enc, Z_dim)
        self.decoder = Decoder(Z_dim, self.IOh_dims_Dec,
                               X_dim, bernoulli, gaussian)
        if (self.encoder.created == False or self.decoder.created == False):
            print("ERROR_VAE: Wrong encoder/decoder structure")
            return None

        # check if NL_types length & layers number are the same
        self.NL_funcE = NL_types_Enc
        self. NL_funcD = NL_types_Dec
        # in Encoder
        if len(self.NL_funcE) != self.encoder.nb_h:
            print("ERROR_VAE: not enough or too many NL functions in encoder")
            return None
        # in Decoder
        if len(self.NL_funcD) != self.decoder.nb_h:
            print("ERROR_VAE: not enough or too many NL functions in decoder")
            return None

        # check if each elemt of NL_types exists in 'torch.nn.functional' module
        # in Encoder
        for index_h in range(self.encoder.nb_h):
            try:
                getattr(F, self.NL_funcE[index_h])
            except AttributeError:
                pass
                print("ERROR_VAE: Wrong encoder NL function name")
                return None
        # in Decoder
        for index_h in range(self.decoder.nb_h):
            try:
                getattr(F, self.NL_funcD[index_h])
            except AttributeError:
                pass
                print("ERROR_VAE: Wrong encoder NL function name")
                return None

        # store encoder and decoder parameters
        self.parameters = []
        for nb_h in range(self.encoder.nb_h):
            self.parameters.append(self.encoder.weights_h[nb_h])
            self.parameters.append(self.encoder.bias_h[nb_h])
        self.parameters.append(self.encoder.weight_mu)
        self.parameters.append(self.encoder.bias_mu)
        self.parameters.append(self.encoder.weight_logSigma)
        self.parameters.append(self.encoder.bias_logSigma)

        for nb_h in range(self.decoder.nb_h):
            self.parameters.append(self.decoder.weights_h[nb_h])
            self.parameters.append(self.decoder.bias_h[nb_h])
        if self.decoder.gaussian and not self.decoder.bernoulli:
            self.parameters.append(self.decoder.weight_mu)
            self.parameters.append(self.decoder.bias_mu)
            self.parameters.append(self.decoder.weight_logSigma)
            self.parameters.append(self.decoder.bias_logSigma)

        # variables to infer
        self.z_mu = None
        self.z_logSigma = None
        self.X_sample = None
        self.X_mu = None
        self.X_logSigma = None

        # minibatch size
        self.mb_size = mb_size
        # learning rate
        self.lr = lr

        # regularization & warm-up
        self.beta = beta
        # avoid zero division
        if Nwu <= 0:
            Nwu = 1
        self.N_wu = Nwu
        self.beta_inc = float(beta) / float(Nwu)
        self.beta_wu = 0

        # VAE training state
        self.epoch_nb = 0
        self.recon_loss = []
        self.regul_loss = []

        self.noise_in = noiseIn
        self.noise_gain = noiseGain

        # flags on vae creation
        self.created = True
        self.trained = False

        # flags on vae state
        self.saved = False
        self.loaded = False

    #---------------------------------------

    def forward(self, X):
        """Compute forward through VAE. Called in trainVAE() for loop (see 'self(X)')."""
        if self.created == False:
            print("ERROR_VAE_forward: VAE not correctly created")
            return None
        # compute z from X
        # if wanted noise input during training
        if (not self.trained) and self.noise_in:
            X = self.noiseInput(X)
        # the size -1 is inferred from other dimensions
        z = self.encode(X.view(-1, self.encoder.dimX))
        # compute X_sample (or X_mu for gaussian) from z
        self.decode(z)

    #---------------------------------------

    def encode(self, X):
        """Encode input with encoder to latent space representation.

        Args:
            X: input data.

        Returns:
            z reparameterized distribution.

        """
        # first layer takes X in input
        var_h = getattr(F, self.NL_funcE[0])(
            torch.mm(X, self.encoder.weights_h[0])
            + self.encoder.bias_h[0].repeat(X.size(0), 1))
        # then variable var_h goes through deeper layers
        for i in range(self.encoder.nb_h - 1):
            var_h = getattr(F, self.NL_funcE[i + 1])(
                torch.mm(var_h, self.encoder.weights_h[i + 1])
                + self.encoder.bias_h[i + 1].repeat(var_h.size(0), 1))

        # get z's mu and logSigma
        self.z_mu = (torch.mm(var_h, self.encoder.weight_mu)
                     + self.encoder.bias_mu.repeat(var_h.size(0), 1))
        self.z_logSigma = (torch.mm(var_h, self.encoder.weight_logSigma)
                           + self.encoder.bias_logSigma.repeat(var_h.size(0), 1))

        # reparametrization trick
        return self.reparameterize()

    #---------------------------------------

    def decode(self, z):
        """Decode latent space update output variables."""
        # first layer takes z in input
        var_h = getattr(F, self.NL_funcD[0])(
            torch.mm(z, self.decoder.weights_h[0])
            + self.decoder.bias_h[0].repeat(z.size(0), 1))
        # then variable var_h goes through deeper layers
        for i in range(self.decoder.nb_h - 1):
            var_h = getattr(F, self.NL_funcD[i + 1])(
                torch.mm(var_h, self.decoder.weights_h[i + 1]) +
                self.decoder.bias_h[i + 1].repeat(var_h.size(0), 1))

        if self.decoder.bernoulli and not self.decoder.gaussian:
            self.X_sample = var_h
        elif self.decoder.gaussian and not self.decoder.bernoulli:
            # get X_sample's mu and logSigma
            self.X_mu = (torch.mm(var_h, self.decoder.weight_mu)
                         + self.decoder.bias_mu.repeat(var_h.size(0), 1))
            self.X_logSigma = (torch.mm(var_h, self.decoder.weight_logSigma)
                               + self.decoder.bias_logSigma.repeat(var_h.size(0), 1))
        else:
            print("ERROR VAE: wrong decoder type")
            raise

    #---------------------------------------

    def reparameterize(self):
        """Reparametrization trick. Enables to compute backward gradient.

        Returns:
            reparameterized distribution

        """
        std = self.z_logSigma.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(self.z_mu)

    #---------------------------------------

    def loss(self, X):
        """Compute loss from reconstrucion (compare input to output)
            and from regularisation (representation)"""

        if self.decoder.bernoulli and not self.decoder.gaussian:
            # Bernoulli
            recon = F.binary_cross_entropy(
                self.X_sample, X.view(-1, self.encoder.dimX))
            recon /= self.mb_size * self.encoder.dimX
        elif self.decoder.gaussian and not self.decoder.bernoulli:
            # Gaussian
            X_sigma = torch.exp(self.X_logSigma)
            firstTerm = torch.log(2 * np.pi * X_sigma)
            secondTerm = ((self.X_mu - X)**2) / X_sigma
            recon = 0.5 * torch.sum(firstTerm + secondTerm)
            recon /= (self.mb_size * self.encoder.dimX)
        else:
            print("ERROR_VAE: VAE type unknown")
            raise
        regul = torch.mean(0.5 * torch.sum(torch.exp(self.z_logSigma)
                                           + self.z_mu**2 - 1.
                                           - self.z_logSigma, 1))
        regul *= self.beta_wu
        loss = recon + regul
        return loss, recon, regul

    #---------------------------------------

    def trainVAE(self, train_loader, epochNb):
        """For loop of (forward -> backward) on entire dataset 

        Args:
            train_loader:   introduces the dataset to the VAE.
            epochNb:        number of loops

        """
        self.train()
        # check mb_size
        if train_loader.batch_size != self.mb_size:
            print("ERROR_VAE_train: batch sizes of data and vae mismatched")
            raise

        optimizer = optim.Adam(self.parameters, self.lr)

        if epochNb <= self.epoch_nb:
            print("ERROR_VAE_train: vae already trained to " +
                  str(self.epoch_nb) + " epochs")
            print("Try a bigger epochs number")
            raise
        for epoch in range(self.epoch_nb + 1, epochNb + 1):

            lossValue = 0
            reconVal = 0
            regulVal = 0

            for i, sample_batched in enumerate(train_loader):

                batch_length = (sample_batched['image'].size(1)
                                * sample_batched['image'].size(0))
                dataset_length = len(train_loader.dataset)

                # make sure the size of the batch corresponds to
                # mbSize*dataSize
                if (batch_length != self.mb_size * self.encoder.dimX):
                    print("ERROR: sizes of data and vae input mismatched")
                    print("batch_length = " + str(batch_length))
                    print("vae input length = " +
                          str(self.mb_size * self.encoder.dimX))
                    raise

                # convert 'double' tensor to 'float' tensor
                X = (sample_batched['image'].view(
                    self.mb_size, self.encoder.dimX)).float()
                X = Variable(X)
                self(X)
                # compute loss between data input and sampled data
                lossVariable, recon, regul = self.loss(X)
                # if nan value, backup VAE before backward
                if math.isnan(lossVariable.data[0]):
                    print("ERROR_VAE_train: there are nan values")
                    self.save('vae_backup_epoch' + str(epoch), './backup')
                    raise
                lossVariable.backward()
                lossValue += lossVariable.data[0]

                # catch recon and regul values
                reconVal += recon.data[0]
                regulVal += regul.data[0]

                optimizer.step()

                # Housekeeping
                for p in self.parameters:
                    if p.grad is not None:
                        data = p.grad.data
                        p.grad = Variable(data.new().resize_as_(data).zero_())

            # gradually increase beta during warm-up
            if(self.beta_wu < self.beta):
                self.beta_wu += self.beta_inc

            # record losse contributions
            self.recon_loss.append(reconVal / dataset_length)
            self.regul_loss.append(regulVal / dataset_length)
            # log out loss current value
            print('====> Epoch: {} Average loss: {:.6f} (recon loss: {:.6f} regul loss: {:.6f})'.format(
                  epoch, lossValue / dataset_length, reconVal / dataset_length, regulVal / dataset_length))

        self.trained = True
        self.epoch_nb = epochNb

    #---------------------------------------

    def save(self, datasetName, saveDir):
        """Save entire VAE class attributes and methods with exhaustive savename"""

        # transform .npz to avoid dot in name (consider only .npz for now)
        name = datasetName.replace(".npz", "_NPZ")
        # add infos on vae structure
        encoderInfo = '_E'
        encoderInfo += '<'
        for numLayerE in range(self.encoder.nb_h):
            encoderInfo += str(self.encoder.inDim_h[numLayerE]) + '-'
            encoderInfo += self.NL_funcE[numLayerE] + '-'
        encoderInfo += str(self.encoder.outDim_h[numLayerE]
                           ) + '-' + 'muSig' + '-'
        encoderInfo += str(self.encoder.dimZ) + '>'
        name += encoderInfo

        decoderInfo = '_D'
        decoderInfo += '<'
        for numLayerD in range(self.decoder.nb_h):
            decoderInfo += str(self.decoder.inDim_h[numLayerD]) + '-'
            decoderInfo += self.NL_funcD[numLayerD] + '-'
        if not self.decoder.bernoulli and self.decoder.gaussian:
            decoderInfo += str(self.decoder.outDim_h[numLayerD]
                               ) + '-' + 'muSig' + '-'
        decoderInfo += str(self.decoder.dimX) + '>'
        name += decoderInfo

        betaInfo = '_beta' + str(self.beta)
        name += betaInfo

        # add infos on training state
        mbSizeInfo = '_mb' + str(self.mb_size)
        name += mbSizeInfo
        lrInfo = '_lr' + str(self.lr).replace(".", "dot")
        name += lrInfo
        epochInfo = '_ep' + str(self.epoch_nb)
        name += epochInfo
        # save it directory
        save_path = saveDir + name

        if not os.path.exists(saveDir):
            os.mkdir(saveDir)

        torch.save(self, save_path)
        print('Saved VAE state into ' + save_path)
        self.saved = True
        return name

    #---------------------------------------

    def getParams(self):
        """Returns all VAE parameters concatenated."""

        listParams = []
        listParams.append(self.encoder.dimX)
        listParams.append(self.decoder.dimZ)
        listParams.append(self.IOh_dims_Enc)
        listParams.append(self.IOh_dims_Dec)
        listParams.append(self.NL_funcE)
        listParams.append(self.NL_funcD)
        listParams.append(self.mb_size)
        listParams.append(self.beta)
        listParams.append(self.lr)
        listParams.append(self.epoch_nb)
        listParams.append(self.decoder.bernoulli)
        listParams.append(self.decoder.gaussian)

        return listParams

    #---------------------------------------

    def noiseInput(self, X):
        """Noise encoder input during training"""
        return X + Variable(self.noise_gain * torch.randn(1, self.decoder.dimX))

    #---------------------------------------

    def generate(self, saveDir, zIndex1, zRange1,  zIndex2, zRange2):
        """Generate samples from decoder. Working only for 2 dimensions of z"""
        if not os.path.exists(saveDir):
            os.mkdir(saveDir)

        self.eval()
        # tensorParamValues = torch.FloatTensor(
        #     frameNb, self.decoder.dimZ).zero_()
        tensorParamValues = torch.FloatTensor(
            2 * zRange2, self.decoder.dimZ).zero_()
        for i in range(-zRange1, zRange1):
            for j in range(2 * zRange2):
                # print(float(i)/float(zRange1))
                tensorParamValues[j][zIndex1] = 10 * float(i) / float(zRange1)
                tensorParamValues[j][zIndex2] = 10 * \
                    float(j - zRange2) / float(zRange2)

            sample = Variable(tensorParamValues)
            self.decode(sample)
            if self.decoder.bernoulli and not self.decoder.gaussian:
                image = self.X_sample.cpu()
            elif self.decoder.gaussian and not self.decoder.bernoulli:
                image = self.X_mu.cpu()
            save_image(image.data.view(2 * zRange2, self.decoder.dimX),
                       saveDir + 'z' + str(zIndex1) +
                       'valOffset' + str(i + zRange1)
                       + '_z' + str(zIndex2) + 'rangeCentered' + str(zRange2) + '.png')

#---------------------------- End class VAE -----------------------------------


#---------------------------- Begin functions ---------------------------------

def loadVAE(vaeSaveName, load_dir):
    """Load a VAE class from a saved file."""

    if not os.path.exists(load_dir):
        print("ERROR_VAE_Load: " + load_dir + " invalid directory")
        raise
    savefile_path = load_dir + vaeSaveName
    if not os.path.exists(savefile_path):
        raise Exception("ERROR_VAE_Load: " + savefile_path + " invalid file")
    vae = torch.load(savefile_path)
    # if vae from gpu
    # vae = torch.load(savefile_path, map_location=lambda storage, loc: storage)

    # check if vae and vaeSameName match
    paramsFromFilename = getParamsFromName(vaeSaveName)
    paramsFromVAE = vae.getParams()
    if paramsFromFilename != paramsFromVAE:
        print("ERROR_LOAD: vae loaded and fileName mismatched")
        raise
    else:
        vae.loaded = True
        return vae

#---------------------------------------


def getParamsFromName(vaeSaveName):
    """Returns all VAE parameters concatenated from parsing saved filename."""
    # e.g.vaeSaveName =
    # 'dummyDataset100_NPZ_E<1024-relu-401-muSig-6>_D<6-relu-399-sigmoid-1024>_beta4_mb10_lr0dot001_ep11'
    s_split = vaeSaveName.split("_")
    datasetName_s = s_split[0]
    datasetType_s = s_split[1]
    encoderNet_s = s_split[2]
    decoderNet_s = s_split[3]
    beta_s = s_split[4]
    mbSize_s = s_split[5]
    lr_s = s_split[6]
    epoch_s = s_split[7]

    # retrieve encoder net dimensions and
    IOh_dims_Enc = []
    NL_types_Enc = []
    # remove labels, only keep values
    encoderNet_s = encoderNet_s.replace(
        "E", "").replace("<", "").replace(">", "")
    encoderNet_tab = encoderNet_s.split("-")
    for i in range(len(encoderNet_tab)):
            # send heaven index val in IOh_dims_Enc
        if i % 2 == 0:
            IOh_dims_Enc.append(int(encoderNet_tab[i]))
        # send odd index val in NL_types_Enc
        else:
            NL_types_Enc.append(encoderNet_tab[i])
    NL_types_Enc.remove('muSig')

    # retrieve decoder net dimensions and
    IOh_dims_Dec = []
    NL_types_Dec = []
    # remove labels, only keep values
    decoderNet_s = decoderNet_s.replace(
        "D", "").replace("<", "").replace(">", "")
    decoderNet_tab = decoderNet_s.split("-")
    for i in range(len(decoderNet_tab)):
        # send heaven index val in IOh_dims_Dec
        if i % 2 == 0:
            IOh_dims_Dec.append(int(decoderNet_tab[i]))
        # send odd index val in NL_types_Dec
        else:
            NL_types_Dec.append(decoderNet_tab[i])
    # check if the decoder is gaussian or bernoulli
    if NL_types_Dec[-1] == 'muSig':
        NL_types_Dec.remove('muSig')
        gaussian = True
        bernoulli = False
    else:
        bernoulli = True
        gaussian = False

    X_dim = IOh_dims_Enc[0]
    Z_dim = IOh_dims_Dec[0]
    beta = int(beta_s.replace("beta", ""))
    mb_size = int(mbSize_s.replace("mb", ""))
    lr = float(lr_s.replace("lr", "").replace("dot", "."))
    epoch_nb = int(epoch_s.replace("ep", ""))

    listParams = []
    listParams.append(X_dim)
    listParams.append(Z_dim)
    listParams.append(IOh_dims_Enc)
    listParams.append(IOh_dims_Dec)
    listParams.append(NL_types_Enc)
    listParams.append(NL_types_Dec)
    listParams.append(mb_size)
    listParams.append(beta)
    listParams.append(lr)
    listParams.append(epoch_nb)
    listParams.append(bernoulli)
    listParams.append(gaussian)

    return listParams

#---------------------------- End functions -----------------------------------
