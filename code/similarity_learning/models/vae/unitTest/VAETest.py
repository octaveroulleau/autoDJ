"""VAETest.py

This module gives unit testing classes for VAE creation, training, generation and save/load.

Todo:
    * add tests on loading
    * add tests on parsing bash command (see mainScript.py)
    * add tests on visualization
"""

import unittest
import sys
import os
import shutil
# Add the src folder path to the sys.path list
sys.path.append('../src')
# sys.path.append('../src/dataset')
sys.path.append('../src/Visualize')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import misc
import numpy
import tensorflow
from tensorflow.examples.tutorials.mnist import input_data
import torch
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms

from VAE import VAE
from VAE import loadVAE
from ManageDataset import NPZ_Dataset
from PCA import PCA_reduction, PCA_vision
from tsne import Hbeta, x2p, tsne


mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)


#---------------------------- Begin class TestVAECreation ----------------

class TestVAECreation(unittest.TestCase):
    """Tests VAE creation."""

    #---------------------------------------

    def test_good_VAE(self):
        X_dim = 513
        Z_dim = 6
        IOh_dims_Enc = [X_dim, 128, Z_dim]
        IOh_dims_Dec = [Z_dim, 128, X_dim]
        NL_types_Enc = ['relu6']
        NL_types_Dec = ['relu6', 'sigmoid']
        model = VAE(X_dim, Z_dim, IOh_dims_Enc,
                    IOh_dims_Dec, NL_types_Enc, NL_types_Dec)
        self.assertTrue(model.created)

    #---------------------------------------

    def test_wrong_EncoderStructure(self):
        X_dim = 513
        Z_dim = 6
        IOh_dims_Enc = [X_dim, Z_dim]
        IOh_dims_Dec = [Z_dim, 128, X_dim]
        NL_types_Enc = ['relu6']
        NL_types_Dec = ['relu6', 'sigmoid']
        model = VAE(X_dim, Z_dim, IOh_dims_Enc,
                    IOh_dims_Dec, NL_types_Enc, NL_types_Dec)
        self.assertFalse(model.created)

    #---------------------------------------

    def test_wrong_DecoderStructure(self):
        X_dim = 513
        Z_dim = 6
        IOh_dims_Enc = [X_dim, 128, Z_dim]
        IOh_dims_Dec = [X_dim]
        NL_types_Enc = ['relu6']
        NL_types_Dec = ['relu6', 'sigmoid']
        model = VAE(X_dim, Z_dim, IOh_dims_Enc,
                    IOh_dims_Dec, NL_types_Enc, NL_types_Dec)
        self.assertFalse(model.created)

    #---------------------------------------

    def test_wrong_EncoderNLFunctionsNb(self):
        X_dim = 513
        Z_dim = 6
        IOh_dims_Enc = [X_dim, 128, Z_dim]
        IOh_dims_Dec = [Z_dim, 128, X_dim]
        NL_types_Enc = ['relu6', 'relu6']
        NL_types_Dec = ['relu6', 'sigmoid']
        model = VAE(X_dim, Z_dim, IOh_dims_Enc,
                    IOh_dims_Dec, NL_types_Enc, NL_types_Dec)
        self.assertFalse(model.created)

    #---------------------------------------

    def test_wrong_DecoderNLFunctionsNb(self):
        X_dim = 513
        Z_dim = 6
        IOh_dims_Enc = [X_dim, 128, Z_dim]
        IOh_dims_Dec = [Z_dim, 128, X_dim]
        NL_types_Enc = ['relu6']
        NL_types_Dec = ['relu6']
        model = VAE(X_dim, Z_dim, IOh_dims_Enc,
                    IOh_dims_Dec, NL_types_Enc, NL_types_Dec)
        self.assertFalse(model.created)

    #---------------------------------------

    def test_wrong_EncoderNLfunctionsSyntax(self):
        X_dim = 513
        Z_dim = 6
        IOh_dims_Enc = [X_dim, 128, Z_dim]
        IOh_dims_Dec = [Z_dim, 128, X_dim]
        NL_types_Enc = ['reLu']
        NL_types_Dec = ['relu6', 'sigmoid']
        model = VAE(X_dim, Z_dim, IOh_dims_Enc,
                    IOh_dims_Dec, NL_types_Enc, NL_types_Dec)
        self.assertFalse(model.created)

    #---------------------------------------

    def test_wrong_DecoderNLfunctionsSyntax(self):
        X_dim = 513
        Z_dim = 6
        IOh_dims_Enc = [X_dim, 128, Z_dim]
        IOh_dims_Dec = [Z_dim, 128, X_dim]
        NL_types_Enc = ['relu6']
        NL_types_Dec = ['relu6', 'sigmoide']
        model = VAE(X_dim, Z_dim, IOh_dims_Enc,
                    IOh_dims_Dec, NL_types_Enc, NL_types_Dec)
        self.assertFalse(model.created)

    #---------------------------------------

    def test_good_gaussianVAE(self):
        X_dim = 513
        Z_dim = 6
        IOh_dims_Enc = [X_dim, 128, Z_dim]
        IOh_dims_Dec = [Z_dim, 128, X_dim]
        NL_types_Enc = ['relu6']
        NL_types_Dec = ['relu6']
        model = VAE(X_dim, Z_dim, IOh_dims_Enc,
                    IOh_dims_Dec, NL_types_Enc, NL_types_Dec, bernoulli=False, gaussian=True)
        self.assertTrue(model.created)

    #---------------------------------------

    def test_wrong_gaussianVAE(self):
        X_dim = 513
        Z_dim = 6
        IOh_dims_Enc = [X_dim, 128, Z_dim]
        IOh_dims_Dec = [Z_dim, 128, X_dim]
        NL_types_Enc = ['relu6']
        NL_types_Dec = ['relu6', 'sigmoid']
        model = VAE(X_dim, Z_dim, IOh_dims_Enc,
                    IOh_dims_Dec, NL_types_Enc, NL_types_Dec, bernoulli=False, gaussian=True)
        self.assertFalse(model.created)

#---------------------------- End class TestVAECreation ------------------

#---------------------------- Begin class TestVAEFunctions ---------------


class TestVAEFunctions(unittest.TestCase):
    """Tests VAE training, saving and loading."""

    #---------------------------------------

    def test_VAE_lonelyForward(self):
        mb_size = 64
        # test on mnist dataset
        X, _ = mnist.train.next_batch(mb_size)
        X = Variable(torch.from_numpy(X))

        # define vae structure
        X_dim = mnist.train.images.shape[1]
        Z_dim = 6
        IOh_dims_Enc = [X_dim, 400, Z_dim]
        IOh_dims_Dec = [Z_dim, 400, X_dim]
        NL_types_Enc = ['relu6']
        NL_types_Dec = ['relu6', 'sigmoid']
        vae = VAE(X_dim, Z_dim, IOh_dims_Enc,
                  IOh_dims_Dec, NL_types_Enc, NL_types_Dec, mb_size)

        optimizer = optim.Adam(vae.parameters, lr=1e-3)
        optimizer.zero_grad()
        vae(X)
        out = vae.X_sample
        vae.encoder.getInfo()
        vae.decoder.getInfo()
        self.assertTrue(vae.created and (
            out.size()[1] == X_dim and out.size()[0] == mb_size))

    #---------------------------------------

    def test_bernoulliVAE_Learning(self):
        mb_size = 1
        # test on mnist dataset
        X, _ = mnist.train.next_batch(mb_size)

        # define vae structure
        X = Variable(torch.from_numpy(X))
        X_dim = mnist.train.images.shape[1]
        Z_dim = 1
        IOh_dims_Enc = [X_dim, 50, Z_dim]
        IOh_dims_Dec = [Z_dim, 50, X_dim]
        NL_types_Enc = ['relu6']
        NL_types_Dec = ['relu6', 'sigmoid']
        vae = VAE(X_dim, Z_dim, IOh_dims_Enc,
                  IOh_dims_Dec, NL_types_Enc, NL_types_Dec, mb_size, bernoulli=True, gaussian=False)

        optimizer = optim.Adam(vae.parameters, lr=1e-3)

        fig = plt.figure()
        ims = []

        for i in range(100):
            optimizer.zero_grad()
            if vae.decoder.gaussian:
                vae(X)
                out = vae.X_mu
            elif vae.decoder.bernoulli:
                vae(X)
                out = vae.X_sample
            else:
                raise
            loss, _, _ = vae.loss(X)
            if i == 0:
                initialLoss = loss.data[0]
            if(i % 10 == 0):
                print("Loss -> " + str(loss.data[0]))
            loss.backward()
            optimizer.step()

            # update plot
            gen = out.data.numpy()
            gen_2D = numpy.reshape(gen[0], (28, 28))
            im = plt.imshow(gen_2D, animated=True)
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                        repeat_delay=1000)

        plt.show()

    #---------------------------------------

    def test_gaussianVAE_Learning(self):
        mb_size = 1
        # test on mnist dataset
        X, _ = mnist.train.next_batch(mb_size)

        # define vae structure
        X = Variable(torch.from_numpy(X))
        X_dim = mnist.train.images.shape[1]
        Z_dim = 1
        IOh_dims_Enc = [X_dim, 50, Z_dim]
        IOh_dims_Dec = [Z_dim, 50, X_dim]
        NL_types_Enc = ['relu6']
        NL_types_Dec = ['relu6']
        vae = VAE(X_dim, Z_dim, IOh_dims_Enc,
                  IOh_dims_Dec, NL_types_Enc, NL_types_Dec, mb_size, bernoulli=False, gaussian=True)

        optimizer = optim.Adam(vae.parameters, lr=1e-3)

        fig = plt.figure()
        ims = []

        for i in range(100):
            optimizer.zero_grad()
            if vae.decoder.gaussian:
                vae(X)
                out = vae.X_mu
            elif vae.decoder.bernoulli:
                vae(X)
                out = vae.X_sample
            else:
                raise
            loss, _, _ = vae.loss(X)
            if i == 0:
                initialLoss = loss.data[0]
            if(i % 10 == 0):
                print("Loss -> " + str(loss.data[0]))
            loss.backward()
            optimizer.step()

            # update plot
            gen = out.data.numpy()
            gen_2D = numpy.reshape(gen[0], (28, 28))
            im = plt.imshow(gen_2D, animated=True)
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                        repeat_delay=1000)

        plt.show()

        self.assertTrue(vae.created and (loss.data[0] < 100))

    #---------------------------------------

    def test_VAE_trainLoop(self):
        mb_size = 49  # because dummyDataset98.npz is a 98 data size
        epoch_nb = 10

        # define dataset
        datasetName = 'dummyDataset98.npz'
        datasetDir = './dummyDataset/'
        testDataset = NPZ_Dataset(datasetName,
                                  datasetDir, 'Spectrums')
        train_loader = torch.utils.data.DataLoader(
            testDataset, batch_size=mb_size, shuffle=True)

        # define vae structure
        X_dim = 1024
        Z_dim = 6
        IOh_dims_Enc = [X_dim, 400, Z_dim]
        IOh_dims_Dec = [Z_dim, 400, X_dim]
        NL_types_Enc = ['relu6']
        NL_types_Dec = ['relu6', 'sigmoid']
        vae = VAE(X_dim, Z_dim, IOh_dims_Enc,
                  IOh_dims_Dec, NL_types_Enc, NL_types_Dec, mb_size)

        vae.trainVAE(train_loader, epoch_nb)
        self.assertTrue(vae.created and vae.trained)

    #---------------------------------------

    def test_VAE_saveState(self):
        mb_size = 49  # because dummyDataset98.npz is a 98 data size
        epoch_nb = 11
        datasetName = 'dummyDataset98.npz'
        datasetDir = './dummyDataset/'
        saveDir = './dummySaveTest/'
        testDataset = NPZ_Dataset(datasetName,
                                  datasetDir, 'Spectrums')
        train_loader = torch.utils.data.DataLoader(
            testDataset, batch_size=mb_size, shuffle=True)

        # define vae structure
        X_dim = 1024
        Z_dim = 6
        IOh_dims_Enc = [X_dim, 401, Z_dim]
        IOh_dims_Dec = [Z_dim, 399, X_dim]
        NL_types_Enc = ['relu6']
        NL_types_Dec = ['relu6', 'sigmoid']
        vae = VAE(X_dim, Z_dim, IOh_dims_Enc,
                  IOh_dims_Dec, NL_types_Enc, NL_types_Dec, mb_size)

        vae.trainVAE(train_loader, epoch_nb)
        # save it
        if vae.trained:
            vae.save(datasetName, saveDir)

        self.assertTrue(vae.created and
                        vae.trained and vae.saved)

    #---------------------------------------

    def test_gaussianVAE_trainsave(self):
        mb_size = 49  # because dummyDataset98.npz is a 98 data size
        epoch_nb = 5
        # if exists remove 'saveloadTest' folder
        if os.path.exists('./saveloadTest'):
            shutil.rmtree('./saveloadTest')
        # create a VAE
        X_dim = 1024
        Z_dim = 10
        IOh_dims_Enc = [X_dim, 600, Z_dim]
        IOh_dims_Dec = [Z_dim, 600, X_dim]
        NL_types_Enc = ['relu6']
        NL_types_Dec = ['relu6']
        vae = VAE(X_dim, Z_dim, IOh_dims_Enc,
                  IOh_dims_Dec, NL_types_Enc, NL_types_Dec, mb_size, bernoulli=False, gaussian=True)
        # prepare dataset
        datasetName = 'dummyDataset98.npz'
        datasetDir = './dummyDataset/'
        saveDir = './dummySaveTest/'
        testDataset = NPZ_Dataset(datasetName,
                                  datasetDir, 'Spectrums')
        train_loader = torch.utils.data.DataLoader(
            testDataset, batch_size=mb_size, shuffle=True)
        # train it for 10 epochs
        vae.trainVAE(train_loader, epoch_nb)
        # save it
        savefile = vae.save(datasetName, saveDir)
        # reload the savefile of VAE
        # vae = loadVAE(savefile, saveDir)

        # continue training
        # vae.trainVAE(train_loader, 10)
        # vae.save(datasetName, saveDir)
        self.assertTrue(vae.created and vae.saved) #and vae.loaded

    def test_VAE_load(self):
        # try to load a vae 
        vae = loadVAE('dummyDataset98_NPZ_E<1024-relu6-600-muSig-10>_D<10-relu6-600-muSig-1024>_beta1_mb49_lr0dot001_ep5',
                      './dummySaveTest/')
        self.assertTrue(vae.created and vae.loaded)

    #---------------------------------------

    # def test_VAE_invalidLoad(self):

    #---------------------------------------

    # def test_VAE_invalidEpochNb(self):

    #---------------------------------------

    # def test_VAE_loadWrongFile(self):

#---------------------------- End class TestVAEFunctions -----------------

#---------------------------- Begin class TestVAEVisualize ---------------


class TestVAEVisualize(unittest.TestCase):
    """Test VAE visualization (PCA and t-SNE)."""

    #---------------------------------------

    def test_VAE_PCA(self):
        # try to load a vae
        vaeLoaded = loadVAE('dummyDataset98_NPZ_E<1024-relu6-401-muSig-6>_D<6-relu6-399-sigmoid-1024>_beta1_mb49_lr0dot001_ep11','./dummySaveTest/')
        test = numpy.load("./dummyDataset/dummyDataset98.npz")
        # vaeLoaded = loadVAE('toy-spectral-richness-v2-lin_NPZ_E<1024-relu6-600-muSig-10>_D<10-relu6-600-muSig-1024>_beta3_mb100_lr0dot001_ep400',
        #                     "../data/savedVAE/Alexis-nonNorm/beta3/WU100/")
        # test = numpy.load("../data/toy-spectral-richness-v2-lin.npz")
        spec = test['Spectrums']
        lbl = test['labels']

        lbltest = []
        lbltest = lbl[:, 0:98:]
        spec = spec[:, 0:98]
        print("~~~ Nb samples :", len(lbltest[0, :]))

        spec = torch.from_numpy(spec)
        spec = spec.float()
        spec = spec.t()
        lblarr = numpy.array(lbltest)

        spec = Variable(spec)

        z_res = vaeLoaded.encode(spec)
        z_res = z_res.data
        z_res = z_res.double()

        PCA_vision(z_res, lblarr, 0)
        plt.figure()

        z_res = z_res.numpy()

        Y = tsne(z_res, 2, 10)
        plt.scatter(Y[:, 0], Y[:, 1], 20, lblarr[0])
        plt.show()

        self.assertTrue(vaeLoaded.loaded)

#---------------------------- End class TestVAEVisualize ------------------

#---------------------------- Test suites ------------------------------------

suiteVAECreation = unittest.TestLoader().loadTestsFromTestCase(TestVAECreation)
print ("\n\n------------------- VAE Creation Test Suite -------------------\n")
unittest.TextTestRunner(verbosity=2).run(suiteVAECreation)
suiteVAEFunctions = unittest.TestLoader().loadTestsFromTestCase(TestVAEFunctions)
print ("\n\n------------------- VAE functions Test Suite -------------------\n")
unittest.TextTestRunner(verbosity=2).run(suiteVAEFunctions)
suiteVAEVisualize = unittest.TestLoader().loadTestsFromTestCase(TestVAEVisualize)
print ("\n\n------------------- VAE visualization Test Suite -------------------\n")
unittest.TextTestRunner(verbosity=2).run(suiteVAEVisualize)
