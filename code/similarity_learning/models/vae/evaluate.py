""" ################# Evaluation mode. ################
2. Using training history, evaluate score on the training dataset (+ validation)
3. Use T-SNE to visually check coherence of the embedding space

@author: laure

"""

import sys
sys.path.append('similarity_learning/models/vae/')
import VAE
import visualize.plot_vae.dimension_reduction as dr
import similarity_learning.models.dielemann.load as cnn_load

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
	

def t_sne_cnn_tasks(data, chunks_list, audioSet):
	"""
	This script evaluates the classification performances of the VAE trained on specific embedding spaces.
	For this, we use the CNN models trained specifically on a task : genre, artist or key recognition.
	We plot a representation of the training data in the latent space using t-sne dimensionality reduction.
	Plots each chunk as a point : one plot for genre, one for artist, one for key.
	"""

	# Load pre-trained CNNs
	cnn_genre = cnn_load.load_CNN_model('genre_full')
	cnn_artist = cnn_load.load_CNN_model('artist_full')
	cnn_key = cnn_load.load_CNN_model('key_full')
	cnn_all = cnn_load.load_CNN_model('genre_full_artist_full_key_full')

	# Perform a forward pass to project to the CNN's embedding spaces
	data_genre = Variable(torch.from_numpy(np.asarray(cnn_genre.predict(data))))
	data_artist = Variable(torch.from_numpy(np.asarray(cnn_artist.predict(data))))
	data_key = Variable(torch.from_numpy(np.asarray(cnn_key.predict(data))))
	data_all = Variable(torch.from_numpy(np.asarray(cnn_all.predict(data))))

	# Load the pre-trained VAEs
	dirpath = 'similarity_learning/models/vae/'
	vae_genre = VAE.load_vae(dirpath + 'saved_models/' + 'vae_genre_full' + '.t7')
	vae_artist = VAE.load_vae(dirpath + 'saved_models/' + 'vae_genre_full' + '.t7') # vae_artist_full
	vae_key = VAE.load_vae(dirpath + 'saved_models/' + 'vae_key_full' + '.t7')
	vae_all = VAE.load_vae(dirpath + 'saved_models/' + 'vae_genre_full_artist_full_key_full' + '.t7')

	# Perform forward pass to project to the VAE's embedding spaces
	print('Projecting to embedding space ...')
	_, _, z_genre  = vae_genre.forward(data_genre)
	embedded_data_genre = z_genre[-1].data.numpy()
	_, _, z_artist  = vae_artist.forward(data_artist)
	embedded_data_artist = z_artist[-1].data.numpy()	
	_, _, z_key  = vae_key.forward(data_key)
	embedded_data_key = z_key[-1].data.numpy()	
	_, _, z_all  = vae_all.forward(data_all)
	embedded_data_all = z_all[-1].data.numpy()

	dim_embedd_space = embedded_data_genre.shape[1]
	nb_chunks_total = embedded_data_genre.shape[0]
	print('Done.')

	# Proceed to dimensionality reduction
	print("TSNE performing ...")
	RS = 20150101
	data_reduced_genre = TSNE(2, random_state=RS).fit_transform(embedded_data_genre)
	data_reduced_artist = TSNE(2, random_state=RS+1).fit_transform(embedded_data_artist)
	data_reduced_key = TSNE(2, random_state=RS-1).fit_transform(embedded_data_key)
	data_reduced_all = TSNE(2, random_state=RS-1).fit_transform(embedded_data_all)
	print("... TSNE performed.")

	# Collect information about each chunk
	track_ids = [ch.track_id for ch in chunks_list.list_of_chunks][:nb_chunks_total]
	genres = [audioSet.metadata["genre"][tid] for tid in track_ids][:nb_chunks_total]
	genres_labels = [[key for key, value in audioSet.classes["genre"].iteritems() if value == i] for i in genres]

	print("check length : ", len(genres) == len(genres_labels))
	print(genres_labels)

	artists = [audioSet.metadata["artist"][tid] for tid in track_ids][:nb_chunks_total]
	artists_labels = [[key for key, value in audioSet.classes["artist"].iteritems() if value == i] for i in artists]
	keys = [audioSet.metadata["key"][tid] for tid in track_ids][:nb_chunks_total]
	keys_labels = [[key for key, value in audioSet.classes["key"].iteritems() if value == i] for i in keys]

	# Create a scatter plot for specialized VAEs.
	f, (a1, a2, a3) = plt.subplots(1, 3)
	a1.scatter(data_reduced_genre[:,0], data_reduced_genre[:,1], c=genres, cmap=plt.cm.spectral, label =genres_labels, edgecolor='k') # alpha = 0.3  
	a1.axis('off')
	a1.axis('tight')
	a1.set_title('VAE with genre recognition CNN')

	a2.scatter(data_reduced_key[:,0], data_reduced_key[:,1], c=keys, cmap=plt.cm.spectral, label=keys_labels, edgecolor='k') # alpha = 0.3  
	a2.axis('off')
	a2.axis('tight')
	a2.set_title('VAE with key recognition CNN')

	a3.scatter(data_reduced_artist[:,0], data_reduced_artist[:,1], c=artists, cmap=plt.cm.spectral, label = artists_labels, edgecolor='k') # alpha = 0.3  
	a3.axis('off')
	a3.axis('tight')
	a3.set_title('VAE with artist recognition CNN')

	# plt.legend()
	# a1.legend() ?
	plt.show()

	# Create a scatter plot for general VAE.
	f, (a1, a2, a3) = plt.subplots(1, 3)
	a1.scatter(data_reduced_all[:,0], data_reduced_all[:,1], c=genres, cmap=plt.cm.spectral, label =genres_labels, edgecolor='k') # alpha = 0.3  
	a1.axis('off')
	a1.axis('tight')
	a1.set_title('VAE with genre recognition CNN')

	a2.scatter(data_reduced_all[:,0], data_reduced_all[:,1], c=keys, cmap=plt.cm.spectral, label=keys_labels, edgecolor='k') # alpha = 0.3  
	a2.axis('off')
	a2.axis('tight')
	a2.set_title('VAE with key recognition CNN')

	a3.scatter(data_reduced_all[:,0], data_reduced_all[:,1], c=artists, cmap=plt.cm.spectral, label = artists_labels, edgecolor='k') # alpha = 0.3  
	a3.axis('off')
	a3.axis('tight')
	a3.set_title('VAE with artist recognition CNN')

	# plt.legend()
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

