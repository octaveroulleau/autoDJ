""" ################# Evaluation mode. ################
1. Defnition of a custom score : distance between two consecutive chunks of a track should be minimal
2. Evaluate score on the training dataset
3. Use T-SNE to visually check coherence of the embedding space

@author: laure

"""

# Use toymix !!! (chunk by chunk)

import sys
sys.path.append('similarity_learning/models/vae/')

import numpy as np
import visualize.plot_vae.dimension_reduction as dr
import VAE

from scipy.spatial.distance import cdist
import torch
from torch.autograd import Variable

import mixing_point as mp

import matplotlib.pyplot as plt

def forward(cnn_data, model):
	""" Input training set in CNN's feature space
	Output the same set in the VAE's latent space
	"""

def score_vae(model):
	"""
	For each track, find the path representing the chunks it is composed of in the latent space
	Cumulate length of all paths
	Print score of the model
	"""

def t_sne_train(model):
	"""
	Plots a representation of the training data in the latent space using t-sne dimensionality reduction.
	Plots each chunk along with its label.
	Plots tracks as paths.

#%% Plotting and dimension reduction
dr.plot_latent2(data, vae, n_points=2000, method=dr.latent_pca, layer=0, write="pca") #, labels=labels, method=dr.latent_tsne (plus lent)
#dr.plot_latent3(dataset, vae, n_points=10000, method=dr.latent_pca, task="class", layer=1)
	"""

def evaluate(data):
	# Load a pre-trained VAE
	filepath = 'similarity_learning/models/vae/saved_models/test_spec_softplus.t7'
	vae = VAE.load_vae(filepath)

	# Perform a forward pass to project data to the embedding space
	x = Variable(torch.from_numpy(data))
	x_params, z_params, z  = vae.forward(x)
	embedded_data = z[-1].data.numpy()
	dim_embedd_space = embedded_data.shape[1]
	nb_chunks_total = embedded_data.shape[0]

	print("OK !")
