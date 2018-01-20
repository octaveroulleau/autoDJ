#################################################################
# Here the file to run to execute the full process in train mode
#################################################################

import tensorflow as tf
import data
from keras.backend.tensorflow_backend import set_session
import numpy as np

import similarity_learning.models.dielemann.load as cnn_load
import pre_processing.load_for_nns as nns_data_load
import similarity_learning.models.vae.train as vae_train

#%% Load data
audioSet, audioOptions = data.import_data.import_data()

#%% Pre-process data
X = nns_data_load.preprocess_for_cnn(audioSet, audioOptions, 100) # len(audioSet.files)

#%% Train CNNs asynchronously + save model

# GPU configuration: check GPU number for training
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.visible_device_list = "2"
set_session(tf.Session(config=config))

#%% Train VAE + save model

model_name = 'genre_full'
model_base, _ = cnn_load.load_CNN_model(model_name)

X_embed = np.asarray(model_base.predict(X, verbose = 1))
vae_train.train_and_save(X_embed, max_epochs= 10)  # max_epochs = 1000
