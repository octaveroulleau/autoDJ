""" ################# Evaluation mode. ################
2. Using training history, evaluate score on the training dataset (+ validation)
3. Use T-SNE to visually check coherence of the embedding space

@author: laure

"""

import sys
sys.path.append('similarity_learning/models/vae/')
import VAE
import visualize.plot_vae.dimension_reduction as dr

import numpy as np
import torch
from torch.autograd import Variable
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle

np.set_printoptions(threshold=np.nan)



def score_vae(model):
	"""
	For each track, find the path representing the chunks it is composed of in the latent space
	Cumulate length of all paths
	Print score of the model
	"""

def t_sne_toymix(data, model_name):
	"""
	Plots a representation of the training data in the latent space using t-sne dimensionality reduction.
	Plots each chunk along with its label.
	Plots tracks as paths.
	"""

	# TODO
	
	# Load a pre-trained VAE
	dirpath = 'similarity_learning/models/vae/'
	vae = VAE.load_vae(dirpath + 'saved_models/' + model_name + '.t7')
	dr.plot_latent2(data, vae, n_points=2000, method=dr.latent_pca, layer=0, write="pca") #, labels=labels, method=dr.latent_tsne (plus lent)
	#dr.plot_latent3(dataset, vae, n_points=10000, method=dr.latent_pca, task="class", layer=1)
	

def t_sne_cnn_tasks(data, model_name, chunks_list, audioSet):
	"""
	Plots a representation of the training data in the latent space using t-sne dimensionality reduction.
	Plots each chunk along with its label : one plot for genre, one for artist, one for key.
	"""

	# Load a pre-trained VAE
	dirpath = 'similarity_learning/models/vae/'
	vae = VAE.load_vae(dirpath + 'saved_models/' + model_name + '.t7')

	# Perform forward pass to project to the embedding space
	x = Variable(torch.from_numpy(data))
	x_params, z_params, z  = vae.forward(x)
	embedded_data = z[-1].data.numpy()
	dim_embedd_space = embedded_data.shape[1]
	nb_chunks_total = embedded_data.shape[0]

	# Proceed to dimensionality reduction
	print("TSNE performing ...")
	RS = 20150101
	data_reduced = TSNE(2, random_state=RS).fit_transform(embedded_data)
	print("... TSNE performed.")

	# Collect information about each chunk
	track_ids = [ch.track_id for ch in chunks_list.list_of_chunks][:nb_chunks_total]
	genres = [audioSet.metadata["genre"][tid] for tid in track_ids][:nb_chunks_total]
	artists = [audioSet.metadata["artist"][tid] for tid in track_ids][:nb_chunks_total]
	keys = [audioSet.metadata["key"][tid] for tid in track_ids][:nb_chunks_total]

	# Create a scatter plot.
	plt.scatter(data_reduced[:,0], data_reduced[:,1], c=artists, cmap=plt.cm.spectral, edgecolor='k') # alpha = 0.3  
	plt.axis('off')
	plt.axis('tight')
	plt.show()
	

def plot_perfs(model_name):

	# Load a pre-trained VAE
	dirpath = 'similarity_learning/models/vae/'
	vae = VAE.load_vae(dirpath + 'saved_models/' + model_name + '.t7')

	log_path = dirpath + 'training_history/' + model_name + '.pkl'
	training_log = pickle.load(open(log_path, 'rb'))

	plt.plot(training_log[0][:15])
	plt.plot(training_log[1][:15])
	plt.legend(["training loss", "validation loss"])
	plt.title("Performances of the VAE with the genre recognition embedding space")
	plt.show()

