# -*- coding: utf-8 -*-

""" Training mode.
1. Get the dataflow from the CNN’s latent space (1 chunk = 1 point + 1 label)
2. Build VAE base architecture
3. Train the VAE unsupervised on this data (1 track = 1 valid path)
4. Adjust the parameters to get a better embedding space
5. Freeze the model (save)
"""


# VAE model

# VAE training
# Task : auto-encoding of the chunks' features (sort of t-sne : organise space), dim 10.
# Alternatively, concatenate 3 chunks and organise a "track space"

# Save obtained vae to saved_models


import sys
import os
import shutil
# Add the src folder path to the sys.path list
sys.path.append('src')
# sys.path.append('src/dataset')
sys.path.append('src/Visualize')

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

from src.VAE import VAE
from src.VAE import loadVAE
from src.ManageDataset import NPZ_Dataset
from src.Visualize.PCA import PCA_reduction, PCA_vision
from src.Visualize.tsne import Hbeta, x2p, tsne

mnist = input_data.read_data_sets('similarity_learning/models/vae/MNIST_data', one_hot=True)

####################### BERNOULLI ###########################

# # good bernoulli vae
# X_dim = 513
# Z_dim = 6
# IOh_dims_Enc = [X_dim, 128, Z_dim]
# IOh_dims_Dec = [Z_dim, 128, X_dim]
# NL_types_Enc = ['relu6']
# NL_types_Dec = ['relu6', 'sigmoid']
# model = VAE(X_dim, Z_dim, IOh_dims_Enc, IOh_dims_Dec, NL_types_Enc, NL_types_Dec)

# #---------------------------------------

# def test_VAE_bernouilli_onlyForward(self):
#     mb_size = 64
#     # test on mnist dataset
#     X, _ = mnist.train.next_batch(mb_size)
#     X = Variable(torch.from_numpy(X))

#     # define vae structure
#     X_dim = mnist.train.images.shape[1]
#     Z_dim = 6
#     IOh_dims_Enc = [X_dim, 400, Z_dim]
#     IOh_dims_Dec = [Z_dim, 400, X_dim]
#     NL_types_Enc = ['relu6']
#     NL_types_Dec = ['relu6', 'sigmoid']
#     vae = VAE(X_dim, Z_dim, IOh_dims_Enc,
#               IOh_dims_Dec, NL_types_Enc, NL_types_Dec, mb_size)

#     optimizer = optim.Adam(vae.parameters, lr=1e-3)
#     optimizer.zero_grad()
#     vae(X)
#     out = vae.X_sample
#     vae.encoder.getInfo()
#     vae.decoder.getInfo()
#     self.assertTrue(vae.created and (
#         out.size()[1] == X_dim and out.size()[0] == mb_size))

# #---------------------------------------

# def test_bernoulliVAE_Learning(self):
#     mb_size = 1
#     # test on mnist dataset
#     X, _ = mnist.train.next_batch(mb_size)

#     # define vae structure
#     X = Variable(torch.from_numpy(X))
#     X_dim = mnist.train.images.shape[1]
#     Z_dim = 1
#     IOh_dims_Enc = [X_dim, 50, Z_dim]
#     IOh_dims_Dec = [Z_dim, 50, X_dim]
#     NL_types_Enc = ['relu6']
#     NL_types_Dec = ['relu6', 'sigmoid']
#     vae = VAE(X_dim, Z_dim, IOh_dims_Enc,
#               IOh_dims_Dec, NL_types_Enc, NL_types_Dec, mb_size, bernoulli=True, gaussian=False)

#     optimizer = optim.Adam(vae.parameters, lr=1e-3)

#     fig = plt.figure()
#     ims = []

#     for i in range(100):
#         optimizer.zero_grad()
#         if vae.decoder.gaussian:
#             vae(X)
#             out = vae.X_mu
#         elif vae.decoder.bernoulli:
#             vae(X)
#             out = vae.X_sample
#         else:
#             raise
#         loss, _, _ = vae.loss(X)
#         if i == 0:
#             initialLoss = loss.data[0]
#         if(i % 10 == 0):
#             print("Loss -> " + str(loss.data[0]))
#         loss.backward()
#         optimizer.step()

#         # update plot
#         gen = out.data.numpy()
#         gen_2D = numpy.reshape(gen[0], (28, 28))
#         im = plt.imshow(gen_2D, animated=True)
#         ims.append([im])

#     ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
#                                     repeat_delay=1000)

#     plt.show()




####################### GAUSSIAN ###########################

## good_gaussianVAE
# X_dim = 513
# Z_dim = 6
# IOh_dims_Enc = [X_dim, 128, Z_dim]
# IOh_dims_Dec = [Z_dim, 128, X_dim]
# NL_types_Enc = ['relu6']
# NL_types_Dec = ['relu6']
# model = VAE(X_dim, Z_dim, IOh_dims_Enc,
#             IOh_dims_Dec, NL_types_Enc, NL_types_Dec, bernoulli=False, gaussian=True)
# self.assertTrue(model.created)


# test on mnist dataset
mb_size = 10
X, _ = mnist.train.next_batch(mb_size)

# define vae structure
X = Variable(torch.from_numpy(X))
X_dim = mnist.train.images.shape[1]
Z_dim = 1
IOh_dims_Enc = [X_dim, 50, Z_dim]
IOh_dims_Dec = [Z_dim, 50, X_dim]
NL_types_Enc = ['relu6']
NL_types_Dec = ['relu6']
vae = VAE(X_dim, Z_dim, IOh_dims_Enc, IOh_dims_Dec, NL_types_Enc, NL_types_Dec, mb_size, bernoulli=False, gaussian=True)

optimizer = optim.Adam(vae.parameters, lr=1e-3)

for i in range(100):
    optimizer.zero_grad()
    vae(X)
    out = vae.X_mu
    loss, _, _ = vae.loss(X)
    if i == 0:
        initialLoss = loss.data[0]
    if(i % 10 == 0):
        print("Loss -> " + str(loss.data[0]))
    loss.backward()
    optimizer.step()

# save it
vae.save('test_train', './')

print("~~~~~~~~~~~~~~~~~~~", vae.created and (loss.data[0] < 100)) # and vae.trained


######### Only for dummyDataset98.npz

# def test_gaussianVAE_trainsave(self):
#     mb_size = 49  # because dummyDataset98.npz is a 98 data size
#     epoch_nb = 5
#     # if exists remove 'saveloadTest' folder
#     if os.path.exists('./saveloadTest'):
#         shutil.rmtree('./saveloadTest')
#     # create a VAE
#     X_dim = 1024
#     Z_dim = 10
#     IOh_dims_Enc = [X_dim, 600, Z_dim]
#     IOh_dims_Dec = [Z_dim, 600, X_dim]
#     NL_types_Enc = ['relu6']
#     NL_types_Dec = ['relu6']
#     vae = VAE(X_dim, Z_dim, IOh_dims_Enc,
#               IOh_dims_Dec, NL_types_Enc, NL_types_Dec, mb_size, bernoulli=False, gaussian=True)
#     # prepare dataset
#     datasetName = 'dummyDataset98.npz'
#     datasetDir = './dummyDataset/'
#     saveDir = './dummySaveTest/'
#     testDataset = NPZ_Dataset(datasetName,
#                               datasetDir, 'Spectrums')
#     train_loader = torch.utils.data.DataLoader(
#         testDataset, batch_size=mb_size, shuffle=True)
#     # train it for 10 epochs
#     vae.trainVAE(train_loader, epoch_nb)
#     # save it
#     savefile = vae.save(datasetName, saveDir)
#     print(vae.created and vae.saved)


# def test_VAE_trainLoop(self):
#     mb_size = 49  # because dummyDataset98.npz is a 98 data size
#     epoch_nb = 10

#     # define dataset
#     datasetName = 'dummyDataset98.npz'
#     datasetDir = './dummyDataset/'
#     testDataset = NPZ_Dataset(datasetName,
#                               datasetDir, 'Spectrums')
#     train_loader = torch.utils.data.DataLoader(
#         testDataset, batch_size=mb_size, shuffle=True)

#     # define vae structure
#     X_dim = 1024
#     Z_dim = 6
#     IOh_dims_Enc = [X_dim, 400, Z_dim]
#     IOh_dims_Dec = [Z_dim, 400, X_dim]
#     NL_types_Enc = ['relu6']
#     NL_types_Dec = ['relu6', 'sigmoid']
#     vae = VAE(X_dim, Z_dim, IOh_dims_Enc,
#               IOh_dims_Dec, NL_types_Enc, NL_types_Dec, mb_size)

#     vae.trainVAE(train_loader, epoch_nb)
#     self.assertTrue(vae.created and vae.trained)

# #---------------------------------------

# def test_VAE_saveState(self):
#     mb_size = 49  # because dummyDataset98.npz is a 98 data size
#     epoch_nb = 11
#     datasetName = 'dummyDataset98.npz'
#     datasetDir = './dummyDataset/'
#     saveDir = './dummySaveTest/'
#     testDataset = NPZ_Dataset(datasetName,
#                               datasetDir, 'Spectrums')
#     train_loader = torch.utils.data.DataLoader(
#         testDataset, batch_size=mb_size, shuffle=True)

#     # define vae structure
#     X_dim = 1024
#     Z_dim = 6
#     IOh_dims_Enc = [X_dim, 401, Z_dim]
#     IOh_dims_Dec = [Z_dim, 399, X_dim]
#     NL_types_Enc = ['relu6']
#     NL_types_Dec = ['relu6', 'sigmoid']
#     vae = VAE(X_dim, Z_dim, IOh_dims_Enc,
#               IOh_dims_Dec, NL_types_Enc, NL_types_Dec, mb_size)

#     vae.trainVAE(train_loader, epoch_nb)
#     # save it
#     if vae.trained:
#         vae.save(datasetName, saveDir)

#     self.assertTrue(vae.created and
#                     vae.trained and vae.saved)













"""
    "train": train a vae from scratch.
    "load": load (from savefile) a vae already trained and visualize/analyse/generate outputs
"""


# parser = argparse.ArgumentParser(description='generic VAE training and saving')


# parser.add_argument('-mode', type=str, default='train', metavar='boolean',
#                     help='mode among "train", "load". (default: "train"):\
#                     \n "train": train a vae from scratch.\
#                     \n "load": load (from savefile) a vae already trained and visualize outputs')
# parser.add_argument('-vae-path', type=str,
#                     default='../unitTest/dummySaveTest/\
#                     dummyDataset98_NPZ_E<1024-relu6-600-muSig-10>\
#                     _D<10-relu6-600-muSig-1024>_beta1_mb49_lr0dot001_ep5')

# # VAE dimensions
# parser.add_argument('-encoderIOdims', nargs='+', type=int, metavar='int list',
#                     help='<Required> IO dimensions of encoder net (example: 1024 600 6)')
# parser.add_argument('-decoderIOdims', nargs='+', type=int, metavar='int list',
#                     help='<Required> IO dimensions of decoder net (example: 6 600 1024)')

# # VAE Non linear functions
# parser.add_argument('-encoderNL', nargs='+', type=str, metavar='string list',
#                     help='<Required> encoder nonlinear activations \
#                     (example: "relu6" for 1 layer or "relu6" "sigmoid" for 2 layers)')
# parser.add_argument('-decoderNL', nargs='+', type=str, metavar='string list',
#                     help='<Required> decoder nonlinear activations \
#                     (example: "relu6" for 1 layer or "relu6" "sigmoid" for 2 layers)')

# # VAE type
# parser.add_argument('-type', type=str, default='gaussian', metavar='bernoulli/gaussian',
#                     help='chose type of vae: either gaussian or bernoulli (default: "gaussian")')

# # load Dataset and save VAE state settings
# parser.add_argument('-dataset-path', type=str, default='../data/dummyDataset98.npz',
#                     metavar='path', help='datasetName.npz file path \
#                     (default: "../data/dummyDataset98.npz")')
# parser.add_argument('-dataKey', type=str, default='images',
#                     metavar='key', help='key for data in .npz dataset (default: "images")')
# parser.add_argument('-save-path', type=str, default='../data/dummySave/',
#                     metavar='path', help='VAE save path after training (default: "../data/dummySave").')

# # training settings
# parser.add_argument('-mb-size', type=int, default=10, metavar='N',
#                     help='input batch size for training (default: 10)')
# parser.add_argument('-epochs', type=int, default=10, metavar='N',
#                     help='number of epochs to train (default: 10)')
# parser.add_argument('-beta', type=int, default=1, metavar='N',
#                     help='beta coefficient for regularization (default: 1)')
# parser.add_argument('-Nwu', type=int, default=1, metavar='N',
#                     help='epochs number for warm-up (default: 1 -> no warm-up)')
# parser.add_argument('-noise', type=float, default=0., metavar='f',
#                     help='noise gain added to data inputs during training (default: 0.)')


# args = parser.parse_args()
# mode = args.mode

# # copy parser args into variables
# mb_size = args.mb_size
# epoch_nb = args.epochs
# beta = args.beta
# Nwu = args.Nwu
# noiseGain = args.noise
# noise = False

# if noiseGain != 0.:
#     noise = True

# # prepare dataset
# datasetName = args.dataset_path.split("/")[-1]
# datasetDir = args.dataset_path.replace(datasetName, "")
# saveDir = args.save_path
# testDataset = NPZ_Dataset(datasetName,
#                           datasetDir, args.dataKey)


# train_loader = torch.utils.data.DataLoader(
#     testDataset, batch_size=mb_size, shuffle=True)

# X_dim = args.encoderIOdims[0]
# Z_dim = args.decoderIOdims[0]
# IOh_dims_Enc = args.encoderIOdims
# IOh_dims_Dec = args.decoderIOdims
# NL_types_Enc = args.encoderNL
# NL_types_Dec = args.decoderNL
# if args.type == 'bernoulli':
#     vae = VAE(X_dim, Z_dim, IOh_dims_Enc,
#               IOh_dims_Dec, NL_types_Enc, NL_types_Dec, mb_size,
#               bernoulli=True, gaussian=False, beta=beta, Nwu=Nwu, noiseIn=noise, noiseGain=noiseGain)
# elif args.type == 'gaussian':
#     vae = VAE(X_dim, Z_dim, IOh_dims_Enc,
#               IOh_dims_Dec, NL_types_Enc, NL_types_Dec, mb_size,
#               bernoulli=False, gaussian=True, beta=beta, Nwu=Nwu, noiseIn=noise, noiseGain=noiseGain)
# else:
#     print("ERROR script: Chose VAE type -> either bernoulli or gaussian")

# # train it for N epochs
# vae.trainVAE(train_loader, epoch_nb)

# # save it
# vae.save(datasetName, saveDir)



