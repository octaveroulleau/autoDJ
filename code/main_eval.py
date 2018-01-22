#####################################################################
# Here the file to run to execute the full process in evaluation mode
####################################################################

import numpy as np
import similarity_learning.models.vae.evaluate as vae_eval

import tensorflow as tf
import data
from keras.backend.tensorflow_backend import set_session
import numpy as np

import similarity_learning.models.dielemann.load as cnn_load
import pre_processing.load_for_nns as nns_data_load
import similarity_learning.models.vae.train as vae_train

import numpy as np
import similarity_learning.models.vae.compose as vae_comp
import similarity_learning.models.vae.evaluate as vae_eval
import pre_processing.chunkify as pr
import re_synthesis.assemble_mixing_points as re
from re_synthesis.const import SR

from similarity_learning.models.vae import VAE

#######################
# DOWNBEATS EVALUATION
#######################

downbeat_evaluation()

#######################
# CNN EVALUATION
#######################

# Feed the data forward in the CNN

# Evaluate the model :
# T-SNE
# MIREVAL

#######################
# VAE EVALUATION
#######################


# Evaluate the model :
# T-SNE
# MIREVAL

#%% Load audio data and pre-process
audioSet, audioOptions = data.import_data.import_data()
chunks_list = pr.dataset_to_chunkList(audioSet, int(SR))
X = nns_data_load.preprocess_for_cnn(audioSet, audioOptions, 120)

#%% Feed the data forward in the CNN
model_name = 'genre_full'
# model_name = 'genre_full_key_full_artist_full'
model_base= cnn_load.load_CNN_model(model_name)
X_embed = np.asarray(model_base.predict(X, verbose = 1))

#%% Feed the data to the VAE and evaluate
model_vae_name = 'vae_genre_full'
# model_vae_name = 'vae_genre_full_key_full_artist_full'
vae_eval.plot_perfs(model_vae_name)
vae_eval.t_sne_cnn_tasks(X_embed, model_vae_name, chunks_list, audioSet)
# vae_eval.t_sne_toymix(X_embed, model_vae_name, chunks_list, audioSet)

######################
# ALL
######################

# Toymix here
