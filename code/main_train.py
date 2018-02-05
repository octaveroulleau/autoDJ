#################################################################
# Here the file to run to execute the full process in train mode
#################################################################

import numpy as np

import data
import similarity_learning.models.dielemann.load as cnn_load
import pre_processing.load_for_nns as nns_data_load
import similarity_learning.models.vae.train as vae_train

# import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import similarity_learning.models.asynchronous.asynchronous as asyn
from similarity_learning.models.asynchronous.asynchronous import asynchronous_learning


#%% Load data
audioSet, audioOptions = data.import_data.import_data()

#%% Pre-process data
X = nns_data_load.preprocess_for_cnn(audioSet, audioOptions, 150) # len(audioSet.files)
print('Dataset loaded.')

#%% Train CNNs asynchronously + save model
# Here an example with the artist recognition task on top of a genre recognition CNN (transfer learning)
transform_type, transform_options = audioSet.getTransforms()
audioSet.files = audioSet.files[0:150]
batch_size = 5
nb_frames = 100
mod_options = {
        'activation': 'relu',
        'batchNormConv': True,
        'FC number': 2048,
        'batchNormDense': True,
        'Alphabet size': 10,
        'Freeze layer': False,
        'batch size': batch_size}
model_name = '_artist_test' 

asyn.asynchronous_learning(audioSet, audioOptions, nb_frames, mod_options, model_name, nb_epochs = 5, batch_size = batch_size,  
                           task = "artist", transfer_learning = True, model_base_name = 'genre_full')


#%% Train VAE + save model
model_name = 'genre_full_artist_test' # complete model
model_cnn = cnn_load.load_CNN_model(model_name)

X_embed = np.asarray(model_cnn.predict(X, verbose = 1))
vae_train.train_and_save(X_embed, 10, 'vae_full_test') 

