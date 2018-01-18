""" Evaluation mode.
1. Defnition of a custom score : distance between two consecutive chunks of a track should be minimal
2. Evaluate score on the training dataset
3. Use T-SNE to visually check coherence of the embedding space
"""

# Use toymix !!! (chunk by chunk)

import sys
sys.path.append('similarity_learning/models/vae/')

import numpy as np
import visualize.plot_vae.dimension_reduction as dr
import VAE

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
	"""


#%% Fake dataset
input_dim = 1000
nb_chunks = 10000
label_dim = 10
data = np.random.rand(nb_chunks,1000).astype('float32')
labels = np.zeros((nb_chunks, label_dim)).astype('int32')

#%% Load a pre-trained VAE
filepath = 'similarity_learning/models/vae/saved_models/test_spec_softplus.t7'
vae = VAE.load_vae(filepath)

#%% Plotting and dimension reduction
dr.plot_latent2(data, vae, n_points=2000, method=dr.latent_pca, layer=0, write="pca") #, labels=labels, method=dr.latent_tsne (plus lent)
#dr.plot_latent3(dataset, vae, n_points=10000, method=dr.latent_pca, task="class", layer=1)