""" Running mode.
"Random walk" in the VAE embedding space (with small cumulated distance)
Returns a list of mixing points : that can be used to re-synthetize a new track
1. Define constraints for composition : number of chunks, probability of switching track
2. Access the embedding space and perform a random walk with pre-defined constraints : track matching
3. From the chunk’s labels returned, create a list of mixing points
"""

# Random walk : draw a line in latent space, discretize, find nearest neighbors.

import numpy as np
from numpy.random import permutation
import torch
from torch.autograd import Variable

import visualize.plot_vae.dimension_reduction as dr
import VAE

#%% Fake dataset
input_dim = 1000
nb_chunks = 123
label_dim = 10
data = np.random.rand(nb_chunks,1000).astype('float32')
labels = np.zeros((nb_chunks, label_dim)).astype('int32')

#%% Load a pre-trained VAE
filepath = 'similarity_learning/models/vae/saved_models/test_spec_softplus.t7'
vae = VAE.load_vae(filepath)

#%% Perform a forward pass to project to embedding space
x = Variable(torch.from_numpy(data))
x_params, z_params, z  = vae.forward(x)
print(z[2])