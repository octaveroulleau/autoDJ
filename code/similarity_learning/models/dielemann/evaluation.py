# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:28:11 2018

@author: octav
"""
import keras
from pre_processing.chunkify import track_to_chunks
from data.sets.audio import DatasetAudio, importAudioData
import skimage.transform as skt
import numpy as np
import os
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
import pdb


def t_sne_one_track(model, model_name, list_ID, audioSet, audioOptions, chosen_ID, frames = 100, Fs = 22050, show_plot = False):
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
	data_out = model.predict(data, verbose = 1)
	x_embed = TSNE().fit_transform(data_out)
	
	#plot t-sne
	if show_plot == True:
		x = []
		y = []
		
		for i in range(len(x_embed)):
			x.append(x_embed[i][0])
			y.append(x_embed[i][1])
				
		plt.scatter(x, y, s = 50, c = meta)
   
		file_dir = './similarity_learning/models/dielemann/figures/'
	
		if not os.path.exists(file_dir):
			os.makedirs(file_dir)
	
		plt.title(model_name + ' showing track ' + str(chosen_ID))
		plt.savefig(file_dir+model_name+'.png')
		plt.show()

def t_sne_multiple_tracks(model, model_name, list_ID, label, audioSet, audioOptions, alphabet_size, frames = 100, Fs = 22050, show_plot = False):
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
			meta.append(chunks[i].get_meta(audioSet,label)) 
			
			#pass to np array and reshape and swap axes
	data = np.array(data)
	meta = np.array(meta)
	
	data = np.swapaxes(np.array(data),1,2)
	
	#predict model outputs
	data_out = model.predict(data, verbose = 1)
	
	x_embed = TSNE().fit_transform(data_out)
	
	#plot t-sne
	
	if show_plot == True:

		x = []
		y = []

		meta_str = list(audioSet.classes[label].keys())
		meta_str = meta_str[1:]
		
		for i in range(len(x_embed)):
			x.append(x_embed[i][0])
			y.append(x_embed[i][1])
				
		plt.scatter(x, y, s = 50, c = meta/alphabet_size)
   
	
		file_dir = './similarity_learning/models/dielemann/figures/'
	
		if not os.path.exists(file_dir):
			os.makedirs(file_dir)
	
		plt.title(label + ' classification for '+model_name)
		plt.savefig(file_dir+model_name+'.png')
		plt.show()
		
def plot_history(model1_name, model2_name, show_plot = False):
	filepath = './similarity_learning/models/dielemann/saved_models/'
	print("Loading training history ...")
	
	history1 = pickle.load(open(filepath+model1_name+'_history.pkl','rb'))
	history2 = pickle.load(open(filepath+model2_name+'_history.pkl','rb'))
	 
	print('Loaded.')
	print('Plotting training history :')
	
	# history_epoch_1 = []
	# val_loss_1 = 0
	# length = 1
	# last_key = ['epoch','0','batch','0']

	# for key in history1.keys():
		# keys_split = keys.split(' ')
		# if last_key[1] != keys_split[1]:
		#     val_loss_1 = val_loss_1/length
		#     history_epoch_1.append(val_loss_1)
		#     val_loss_1 = 0
		#     length = 1
		# val_loss_1 = val_loss_1 + history1[keys]['val_loss'][0]
		# length = length + 1
		# last_key = keys_split
	
	# history_epoch_2 = []
	# val_loss_2 = 0
	# length = 1
	# last_key = ['epoch','0','batch','0']
	# for keys in history2.keys():
		# keys_split = keys.split(' ')
		# if last_key[1] != keys_split[1]:
		#     val_loss_2 = val_loss_2/length
		#     history_epoch_2.append(val_loss_2)
		#     val_loss_1 = 0
		#     length = 1
		# val_loss_2 = val_loss_2 + history1[keys]['val_loss'][0]
		# length = length + 1
		# last_key = keys_split


	acc1 = [0]*22
	lens = [0]*22
	for key in history1.keys():
	   keys_split = key.split(' ')
	   epoch = int(keys_split[1])
	   lens[epoch] = lens[epoch] + 1
	   acc1[epoch] = acc1[epoch] + history1[key]['val_acc'][0]

	acc2 = [0]*20
	lens = [0]*20
	for key in history2.keys():
		keys_split = key.split(' ')
		epoch = int(keys_split[1])
		lens[epoch] = lens[epoch] + 1
		acc2[epoch] = acc2[epoch] + history2[key]['val_acc'][0]


	history_epoch_1 = [acc1[i]/lens[i] for i in range(len(acc2))]
	history_epoch_2 = [acc2[i]/lens[i] for i in range(len(acc2))]
	
	if show_plot == True:
		file_dir = './similarity_learning/models/dielemann/figures/history/'
	
		if not os.path.exists(file_dir):
			os.makedirs(file_dir)
			
		plt.plot(range(len(history_epoch_1)), history_epoch_1, label = model1_name)
		plt.plot(range(len(history_epoch_2)), history_epoch_2, label = model2_name)
		plt.title('Comparison of the performances of two CNNs')
		plt.legend()
	
		plt.title('Validation accuracy : ' +model1_name+'/'+model2_name+' comparison')
		plt.savefig(file_dir+model1_name+model2_name+'.png')

		plt.show()

				
				
		
		
		


		
	
	
	
	