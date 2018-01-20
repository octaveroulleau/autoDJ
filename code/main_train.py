#################################################################
# Here the file to run to execute the full process in train mode
#################################################################

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import numpy as np

import data
from data.sets.audio import DatasetAudio
import similarity_learning.models.dielemann.load as cnn_load
import pre_processing.load_for_nns as nns_data_load
import similarity_learning.models.vae.train as vae_train

#%% Load data
audioSet, audioOptions = data.import_data.import_data()

#%% Pre-process data
X = nns_data_load.preprocess_for_cnn(audioSet, audioOptions, len(audioSet.files)) # len(audioSet.files)
print('Dataset loaded.')

#%% Train CNNs asynchronously + save model

#%% Train VAE + save model

model_name = 'genre_full'
model_base, _ = cnn_load.load_CNN_model(model_name)

X_embed = np.asarray(model_base.predict(X, verbose = 1))
vae_train.train_and_save(X_embed, 20, 'vae_genre_full')  # max_epochs = 1000
