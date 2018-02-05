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
from pre_processing.chunkify import track_to_chunks

import numpy as np
import torch
from torch.autograd import Variable
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import skimage.transform as skt
import matplotlib.pyplot as plt
import pickle

np.set_printoptions(threshold=np.nan)

def t_sne_cnn_tasks(data, chunks_list, audioSet):
	"""
	This script evaluates the classification performances of the VAE trained on specific embedding spaces.
	For this, we use the CNN models trained specifically on a task : genre, artist or key recognition.
	We plot a representation of the training data in the latent space using t-sne dimensionality reduction.
	Plots each chunk as a point : one plot for genre, one for artist, one for key.
	"""

	######################## 1/ Load all VAEs and CNNs

	# # Load pre-trained CNNs
	# cnn_genre = cnn_load.load_CNN_model('genre_full')
	# cnn_artist = cnn_load.load_CNN_model('artist_full')
	# cnn_key = cnn_load.load_CNN_model('key_full')
	# cnn_all = cnn_load.load_CNN_model('genre_full_artist_full_key_full')

	# # Load the pre-trained VAEs
	# dirpath = 'similarity_learning/models/vae/'
	# vae_genre = VAE.load_vae(dirpath + 'saved_models/' + 'vae_genre_full' + '.t7')
	# vae_artist = VAE.load_vae(dirpath + 'saved_models/' + 'vae_artist_full' + '.t7')
	# vae_key = VAE.load_vae(dirpath + 'saved_models/' + 'vae_key_full' + '.t7')
	# vae_all = VAE.load_vae(dirpath + 'saved_models/' + 'vae_genre_full_artist_full_key_full' + '.t7')
	
	# # Perform a forward pass to project to the CNN's embedding spaces
	# print('Projecting to embedding space ...')
	# data_genre = Variable(torch.from_numpy(np.asarray(cnn_genre.predict(data))))
	# data_artist = Variable(torch.from_numpy(np.asarray(cnn_artist.predict(data))))
	# data_key = Variable(torch.from_numpy(np.asarray(cnn_key.predict(data))))
	# data_all = Variable(torch.from_numpy(np.asarray(cnn_all.predict(data))))

	# # Perform forward pass to project to the VAE's embedding spaces
	# _, _, z_genre  = vae_genre.forward(data_genre)
	# embedded_data_genre = z_genre[-1].data.numpy()
	# _, _, z_artist  = vae_artist.forward(data_artist)
	# embedded_data_artist = z_artist[-1].data.numpy()	
	# _, _, z_key  = vae_key.forward(data_key)
	# embedded_data_key = z_key[-1].data.numpy()	
	# _, _, z_all  = vae_all.forward(data_all)
	# embedded_data_all = z_all[-1].data.numpy()

	# dim_embedd_space = embedded_data_genre.shape[1]
	# nb_chunks_total = embedded_data_genre.shape[0]
	# print('Done.')

	# # Proceed to dimensionality reduction
	# print("TSNE performing ...")

	# data_reduced_genre = TSNE(2).fit_transform(embedded_data_genre)
	# data_reduced_artist = TSNE(2).fit_transform(embedded_data_artist)
	# data_reduced_key = TSNE(2).fit_transform(embedded_data_key)
	# data_reduced_all = TSNE(2).fit_transform(embedded_data_all)
	# print("... TSNE performed.")

	# # Collect information about each chunk
	# track_ids = [ch.track_id for ch in chunks_list.list_of_chunks][:nb_chunks_total]

	# genres = [audioSet.metadata["genre"][tid] for tid in track_ids][:nb_chunks_total]
	# g = sorted(list(set(genres)))
	# genres_labels = [[key for key, value in audioSet.classes["genre"].items() if value == i] for i in g]
	# artists = [audioSet.metadata["artist"][tid] for tid in track_ids][:nb_chunks_total]
	# a = sorted(list(set(artists)))
	# artists_labels = [[key for key, value in audioSet.classes["artist"].items() if value == i] for i in a]
	# keys = [audioSet.metadata["key"][tid] for tid in track_ids][:nb_chunks_total]
	# k = sorted(list(set(keys)))
	# keys_labels = [[key for key, value in audioSet.classes["key"].items() if value == i] for i in k]


	# #################################### 2/ Plot one track on one VAE
	# Load pre-trained CNN
	cnn_all = cnn_load.load_CNN_model('genre_full_artist_full_key_full')
	# Load the pre-trained VAE
	dirpath = 'similarity_learning/models/vae/'
	vae_name = 'vae_full_beta1_1layer800'
	vae_all = VAE.load_vae(dirpath + 'saved_models/' + vae_name + '.t7')

	# Perform a forward pass to project to the CNN's embedding spaces
	print('Projecting to embedding space ...')
	data_all = Variable(torch.from_numpy(np.asarray(cnn_all.predict(data))))
	_, _, z_all  = vae_all.forward(data_all)
	embedded_data_all = z_all[-1].data.numpy()
	dim_embedd_space = embedded_data_all.shape[1]
	nb_chunks_total = embedded_data_all.shape[0]
	print('Done.')

	# Proceed to dimensionality reduction
	print("TSNE performing ...")
	data_reduced_all = TSNE(2).fit_transform(embedded_data_all)
	print("... TSNE performed.")
	
	track_ids = [ch.track_id for ch in chunks_list.list_of_chunks][:nb_chunks_total]
	c = [20 if track_ids[i] == 8 else 350 for i in range(len(track_ids))]
	s = [400 - x for x in c]
	plt.scatter(data_reduced_all[:,0], data_reduced_all[:,1], c=c, s=s, cmap=plt.cm.viridis, edgecolor='k', alpha = 0.8) # alpha = 0.3  
	plt.title('VAE' + vae_name + 'showing one track')
	plt.show()

	#################################### 2b/ Plot one track on several VAEs
	# # Load pre-trained CNN
	# cnn_all = cnn_load.load_CNN_model('genre_full_artist_full_key_full')
	# # Load the pre-trained VAE
	# dirpath = 'similarity_learning/models/vae/'
	# vae_1 = VAE.load_vae(dirpath + 'saved_models/' + 'vae_full_beta4_2layers' + '.t7')
	# vae_2 = VAE.load_vae(dirpath + 'saved_models/' + 'vae_full_beta1_1layer800' + '.t7')
	# vae_3 = VAE.load_vae(dirpath + 'saved_models/' + 'vae_full_beta4_1layer100' + '.t7')

	# # Perform a forward pass to project to the CNN's embedding spaces
	# print('Projecting to embedding space ...')
	# data_all = Variable(torch.from_numpy(np.asarray(cnn_all.predict(data))))
	# _, _, z_1  = vae_1.forward(data_all)
	# embedded_data_1 = z_1[-1].data.numpy()
	# _, _, z_2  = vae_2.forward(data_all)
	# embedded_data_2 = z_2[-1].data.numpy()
	# _, _, z_3  = vae_3.forward(data_all)
	# embedded_data_3 = z_3[-1].data.numpy()
	# nb_chunks_total = embedded_data_1.shape[0]
	# print('Done.')

	# # Proceed to dimensionality reduction
	# print("TSNE performing ...")
	# data_reduced_1 = TSNE(2).fit_transform(embedded_data_1)
	# data_reduced_2 = TSNE(2).fit_transform(embedded_data_2)
	# data_reduced_3 = TSNE(2).fit_transform(embedded_data_3)
	# print("... TSNE performed.")

	# track_ids = [ch.track_id for ch in chunks_list.list_of_chunks][:nb_chunks_total]
	# c = [20 if track_ids[i] == 17 else 350 for i in range(len(track_ids))]
	# s = [400 - x for x in c]
	# f, (a1, a2, a3) = plt.subplots(1, 3)
	# a1.scatter(data_reduced_1[:,0], data_reduced_1[:,1], c=c, cmap=plt.cm.viridis, edgecolor='k', alpha = 0.5) # alpha = 0.3
	# a2.scatter(data_reduced_2[:,0], data_reduced_2[:,1], c=c, cmap=plt.cm.viridis, edgecolor='k', alpha = 0.5) # alpha = 0.3
	# a3.scatter(data_reduced_3[:,0], data_reduced_3[:,1], c=c, cmap=plt.cm.viridis, edgecolor='k', alpha = 0.5) # alpha = 0.3  , label = artists_labels
	# a1.axis('off')
	# a1.axis('tight')
	# a1.set_title('One track')
	# a2.axis('off')
	# a2.axis('tight')
	# a2.set_title('One track')
	# a3.axis('off')
	# a3.axis('tight')
	# a3.set_title('One track')
	# plt.show()

	################################## 3/ To print one track with several VAEs
	#%% To print one track
	# f, ((a1, a2), (a3, a4)) = plt.subplots(2, 2)
	# c = [20 if track_ids[i] == 8 else 350 for i in range(len(track_ids))]
	# s = [400 - x for x in c]
	# a1.scatter(data_reduced_genre[:,0], data_reduced_genre[:,1], c=c, s=s, cmap=plt.cm.viridis,edgecolor='k', alpha = 0.8) # alpha = 0.3  
	# a2.scatter(data_reduced_key[:,0], data_reduced_key[:,1], c=c, s=s, cmap=plt.cm.viridis, edgecolor='k', alpha = 0.8) # alpha = 0.3  
	# a3.scatter(data_reduced_artist[:,0], data_reduced_artist[:,1], s=s, c=c, cmap=plt.cm.viridis, edgecolor='k', alpha = 0.8) # alpha = 0.3  
	# a4.scatter(data_reduced_all[:,0], data_reduced_all[:,1], c=c, s=s, cmap=plt.cm.viridis, edgecolor='k', alpha = 0.8) # alpha = 0.3  
	# plt.show()

	# a1.axis('off')
	# a1.axis('tight')
	# a1.set_title('Genre recognition VAE')
	# a2.axis('off')
	# a2.axis('tight')
	# a2.set_title('Key recognition VAE')
	# a3.axis('off')
	# a3.axis('tight')
	# a3.set_title('Artist recognition VAE')
	# a4.axis('off')
	# a4.axis('tight')
	# a4.set_title('Full VAE')
	# plt.show()

	################################# 4/ To print the tasks on several VAEs 

	# l = min(len(data_reduced_genre[:,0]), len(genres))

	# # A. Specialized VAEs on their task
	# f, (a1, a2, a3) = plt.subplots(1, 3)
	# a1.scatter(data_reduced_genre[:,0], [data_reduced_genre[:l,1]], c=genres[:l], cmap=plt.cm.viridis, edgecolor='k', alpha = 0.5) 
	# a2.scatter(data_reduced_key[:l,0], data_reduced_key[:l,1], c=keys[:l], cmap=plt.cm.viridis, edgecolor='k', alpha = 0.5)
	# a3.scatter(data_reduced_artist[:l,0], data_reduced_artist[:l,1], c=artists[:l], cmap=plt.cm.viridis, edgecolor='k', alpha = 0.5) 

	# a1.axis('off')
	# a1.axis('tight')
	# a1.set_title('VAE with genre recognition CNN')
	# a2.axis('off')
	# a2.axis('tight')
	# a2.set_title('VAE with key recognition CNN')
	# a3.axis('off')
	# a3.axis('tight')
	# a3.set_title('VAE with artist recognition CNN')
	# plt.show()

	# # B. General VAE on all tasks.
	# f, (a1, a2, a3) = plt.subplots(1, 3)

	# a1.scatter(data_reduced_all[:l,0], data_reduced_all[:l,1], c=genres[:l], cmap=plt.cm.viridis, edgecolor='k', alpha = 0.5) # alpha = 0.3
	# a2.scatter(data_reduced_all[:l,0], data_reduced_all[:l,1], c=keys[:l], cmap=plt.cm.viridis, edgecolor='k', alpha = 0.5) # alpha = 0.3
	# a3.scatter(data_reduced_all[:l,0], data_reduced_all[:l,1], c=artists[:l], cmap=plt.cm.viridis, edgecolor='k', alpha = 0.5) # alpha = 0.3  , label = artists_labels

	# a1.axis('off')
	# a1.axis('tight')
	# a1.set_title('Full VAE on genre')
	# a2.axis('off')
	# a2.axis('tight')
	# a2.set_title('Full VAE on key')
	# a3.axis('off')
	# a3.axis('tight')
	# a3.set_title('Full VAE on artist')
	# plt.show()

	################################# 4b/ To print the tasks on a given VAE

	# # Load pre-trained CNN
	# cnn_all = cnn_load.load_CNN_model('genre_full_artist_full_key_full')
	# # Load the pre-trained VAE
	# dirpath = 'similarity_learning/models/vae/'
	# vae_all = VAE.load_vae(dirpath + 'saved_models/' + 'vae_full_beta4_2layers' + '.t7')
	# # 'vae_full_beta1_1layer800' vae_full_beta4_1layer100 vae_full_beta4_2layers

	# # Perform a forward pass to project to the CNN's embedding spaces
	# print('Projecting to embedding space ...')
	# data_all = Variable(torch.from_numpy(np.asarray(cnn_all.predict(data))))
	# _, _, z_all  = vae_all.forward(data_all)
	# embedded_data_all = z_all[-1].data.numpy()
	# dim_embedd_space = embedded_data_all.shape[1]
	# nb_chunks_total = embedded_data_all.shape[0]
	# print('Done.')

	# # Proceed to dimensionality reduction
	# print("TSNE performing ...")
	# data_reduced_all = TSNE(2).fit_transform(embedded_data_all)
	# print("... TSNE performed.")

	# # Collect information about each chunk
	# track_ids = [ch.track_id for ch in chunks_list.list_of_chunks][:nb_chunks_total]
	# genres = [audioSet.metadata["genre"][tid] for tid in track_ids][:nb_chunks_total]
	# artists = [audioSet.metadata["artist"][tid] for tid in track_ids][:nb_chunks_total]
	# keys = [audioSet.metadata["key"][tid] for tid in track_ids][:nb_chunks_total]

	# f, (a1, a2, a3) = plt.subplots(1, 3)

	# a1.scatter(data_reduced_all[:,0], data_reduced_all[:,1], c=genres, cmap=plt.cm.viridis, edgecolor='k', alpha = 0.5) # alpha = 0.3
	# a2.scatter(data_reduced_all[:,0], data_reduced_all[:,1], c=keys, cmap=plt.cm.viridis, edgecolor='k', alpha = 0.5) # alpha = 0.3
	# a3.scatter(data_reduced_all[:,0], data_reduced_all[:,1], c=artists, cmap=plt.cm.viridis, edgecolor='k', alpha = 0.5) # alpha = 0.3  , label = artists_labels

	# a1.axis('off')
	# a1.axis('tight')
	# a1.set_title('Full VAE on genre')
	# a2.axis('off')
	# a2.axis('tight')
	# a2.set_title('Full VAE on key')
	# a3.axis('off')
	# a3.axis('tight')
	# a3.set_title('Full VAE on artist')
	# plt.show()

	# ################################# 5/ To print the tasks on several VAEs WITH LABELS

	# # A. Specialized VAEs on their task
	# f, (a1, a2, a3) = plt.subplots(1, 3)
	# c = ['skyblue', 'lightgray', 'chartreuse', 'salmon', 'cadetblue', 'steelblue', 'honeydew', 'indigo', 'lemonchiffon', 'burlywood']
	
	# i = 0
	# for gi in g :
	# 	condition = [g == gi for g in genres]
	# 	genre0 = data_reduced_genre[condition, 0]
	# 	genre1 = data_reduced_genre[condition, 1]
	# 	a1.scatter(genre0, genre1, c=c[i % 10], cmap=plt.cm.viridis, label = genres_labels[i], edgecolor='k', alpha = 0.5) # alpha = 0.3 
	# 	i = i+1
	# i = 0
	# for ai in a :
	# 	condition = [a == ai for a in artists]
	# 	artist0 = data_reduced_artist[condition, 0]
	# 	artist1 = data_reduced_artist[condition, 1]
	# 	a2.scatter(artist0, artist1, c=c[i % 10], cmap=plt.cm.viridis, edgecolor='k', alpha = 0.5) # alpha = 0.3 
	# 	i = i+1
	# i = 0
	# for ki in k :
	# 	condition = [k == ki for k in keys]
	# 	key0 = data_reduced_key[condition, 0]
	# 	key1 = data_reduced_key[condition, 1]
	# 	a3.scatter(key0, key1, c=c[i % 10], cmap=plt.cm.viridis, label = keys_labels[i], edgecolor='k', alpha = 0.5) # alpha = 0.3 
	# 	i = i+1

	# a1.axis('off')
	# a1.axis('tight')
	# a1.set_title('VAE with genre recognition CNN')
	# a1.legend()
	# a2.axis('off')
	# a2.axis('tight')
	# a2.set_title('VAE with key recognition CNN')
	# a2.legend()
	# a3.axis('off')
	# a3.axis('tight')
	# a3.set_title('VAE with artist recognition CNN')
	# a3.legend()
	# plt.show()

	# # B. General VAE on all tasks.
	# f, (a1, a2, a3) = plt.subplots(1, 3)

	# i = 0
	# for gi in g :
	# 	condition = [g == gi for g in genres]
	# 	genre0 = data_reduced_all[condition, 0]
	# 	genre1 = data_reduced_all[condition, 1]
	# 	a1.scatter(genre0, genre1, c=c[i % 10], cmap=plt.cm.viridis, label = genres_labels[i], edgecolor='k', alpha = 0.5) # alpha = 0.3 		
	# 	i = i+1
	# i = 0
	# for ai in a :
	# 	condition = [a == ai for a in artists]
	# 	artist0 = data_reduced_all[condition, 0]
	# 	artist1 = data_reduced_all[condition, 1]
	# 	a2.scatter(artist0, artist1, c=c[i % 10], cmap=plt.cm.viridis, edgecolor='k', alpha = 0.5) # alpha = 0.3 
	# 	i = i+1
	# i = 0
	# for ki in k :
	# 	condition = [k == ki for k in keys]
	# 	key0 = data_reduced_all[condition, 0]
	# 	key1 = data_reduced_all[condition, 1]
	# 	a3.scatter(key0, key1, c=c[i % 10], cmap=plt.cm.viridis, label = keys_labels[i], edgecolor='k', alpha = 0.5) # alpha = 0.3 
	# 	i = i+1

	# a1.axis('off')
	# a1.axis('tight')
	# a1.set_title('Full VAE on genre')
	# a1.legend()
	# a2.axis('off')
	# a2.axis('tight')
	# a2.set_title('Full VAE on key')
	# a2.legend()
	# a3.axis('off')
	# a3.axis('tight')
	# a3.set_title('Full VAE on artist')
	# a3.legend()
	# plt.show()

def tsne_tracks(model_cnn, list_ID, chosen_ID, audioSet, audioOptions, alphabet_size, frames = 100, Fs = 22050):
	data = []
	meta = []

	for test_ID in list_ID:
		#import track and label
		downbeat = audioSet.metadata['downbeat'][test_ID][0]
		
		#track to chunk
		chunks = track_to_chunks(test_ID, Fs, downbeat)
		
		#for each chunk
		#chunk.get_cqt and resample
		for i in range(len(chunks)):
			chunk = chunks[i].get_cqt(audioSet, audioOptions)
			nbBins = chunk.shape[0]
			chunk = skt.resize(chunk, (nbBins, frames), mode='reflect')
			data.append(chunk)
			if test_ID == chosen_ID :
				meta.append(0)
			else:
				meta.append(1)
			
	#pass to np array and reshape and swap axes
	data = np.array(data)
	meta = np.array(meta)
	data = np.swapaxes(np.array(data),1,2)
	
	#predict model outputs
	dirpath = 'similarity_learning/models/vae/'
	model_vae_name = 'vae_full_beta1_1layer800'
	vae_model = VAE.load_vae(dirpath + 'saved_models/' + model_vae_name + '.t7')
	print('Projecting to embedding space ...')
	data_out = Variable(torch.from_numpy(np.asarray(model_cnn.predict(data, verbose = 1))))
	_, _, z_out  = vae_model.forward(data_out)
	embedded_data = z_out[-1].data.numpy()
	nb_chunks_total = embedded_data.shape[0]
	print('Done.')

	# Proceed to dimensionality reduction
	print("TSNE performing ...")
	x_embed = TSNE(2).fit_transform(embedded_data)
	print('Done.')
	
	#plot t-sne
	x = []
	y = []
	
	for i in range(len(x_embed)):
		x.append(x_embed[i][0])
		y.append(x_embed[i][1])
			
	plt.scatter(x, y, s = 50, c = meta/alphabet_size)
	file_dir = './similarity_learning/models/dielemann/figures/'
	plt.title(model_vae_name + ' showing tracks ' + str(chosen_ID))
	plt.savefig(file_dir+model_vae_name+'.png')
	plt.show()



def plot_perfs(model_name):

	dirpath = 'similarity_learning/models/vae/'
	log_path = dirpath + 'training_history/' + model_name + '.pkl'
	file = open(log_path, 'rb')
	training_log = pickle.load(file)
	file.close()

	plt.plot(training_log[0][:15])
	plt.plot(training_log[1][:15])
	plt.legend(["training loss", "validation loss"])
	plt.title("Performances of the VAE " + model_name)
	plt.show()



