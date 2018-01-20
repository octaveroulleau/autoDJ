# -*- coding: utf-8 -*-

""" ########################## Training mode. #######################
1. Get the dataflow from the CNN’s latent space (1 chunk = 1 point in dim 512)
2. Build VAE base architecture
3. Train the VAE unsupervised on this data (1 track = 1 valid path)
4. Adjust the parameters to get a better embedding space
5. Freeze the model (save)

@author: laure

"""

# Task : auto-encoding of the chunks' features (sort of t-sne : organise space), dim 10.
# TODO : Alternatively, concatenate 3 chunks and organise a "track space"

import sys
sys.path.append('similarity_learning/models/vae/')
import VAE

import numpy as np
import pickle

def train_and_save(data, max_epochs):
	""" Trains the VAE model on the given dataset and saves it as a .t7 serialized object.

	Parameters
    ----------
    data : 2D numpy array of size nb_chunks x dim_data
    	The matrix of the training data vectors

    Returns
    -------
	None. Saves the trained model to disk in the similarity_learning/models/vae/saved_models/ folder.

    Example
    -------

	data = np.random.rand(nb_chunks,1000).astype('float32')
	train_and_save(data)

	"""
	input_dim = data.shape[1]
	nb_chunks = data.shape[0]

	#%% Make model
	model_type="dlgm"
	_, vae = VAE.build_model(model_type, input_dim)

	#%% Defining optimizer
	trainingOptions = {'lr':1e-4}
	vae.init_optimizer(usePyro=False, optimArgs=trainingOptions)

	use_cuda = False
	if use_cuda:
	    vae.cuda()

	#%% Define training setting and train
	batch_size=100
	vae, logs = VAE.train_vae(vae, data, max_epochs=max_epochs, batch_size=batch_size, model_type=model_type, use_cuda=use_cuda)

	#%% Save model to saved_models once trained
	test_name = "test_vae_cnn_1"
	save_dir = 'similarity_learning/models/vae/saved_models/'
	vae.save(save_dir + test_name + '.t7')
	print("Saved " + model_type + " VAE to " + save_dir + test_name + ".")

	#%% Save training history as well
	save_dir = 'similarity_learning/models/vae/training_history/'
	logs_name = save_dir + test_name + '.pkl'
	pickle.dump(logs, open(logs_name, 'wb'))
	print("Saved training history to " + logs_name)





