#####################################################################
# Here the file to run to execute the full process in evaluation mode
####################################################################

import numpy as np

import data
import similarity_learning.models.vae.evaluate as vae_eval
import similarity_learning.models.dielemann.load as cnn_load
import beat_detection.downbeat as beat
import pre_processing.load_for_nns as nns_data_load
import pre_processing.chunkify as pr
from re_synthesis.const import SR

import torch
import tensorflow as tf
import keras
import data
from similarity_learning.models.dielemann.build import GlobalLPPooling1D
from similarity_learning.models.dielemann.evaluation import t_sne_multiple_tracks, plot_history, t_sne_one_track
 

#%% Load audio data and pre-process
audioSet, audioOptions = data.import_data.import_data()
chunks_list = pr.dataset_to_chunkList(audioSet, int(SR))
X = nns_data_load.preprocess_for_cnn(audioSet, audioOptions, 150) # len(audioSet.files)

#######################
# DOWNBEATS EVALUATION
#######################

beat.downbeat_evaluation(audioSet.files, audioSet.metadata["downbeat"])

# #######################
# # CNN EVALUATION
# #######################

model_name = 'genre_full'
model_genre = cnn_load.load_CNN_model(model_name, base_dir = './similarity_learning/models/dielemann/saved_models/', model_type = 'base')
t_sne_multiple_tracks(model_genre, model_name, [0, 100, 200, 300, 400, 500, 600 ,700 ,800, 900], 'genre', audioSet, audioOptions, alphabet_size =10, show_plot = True)
t_sne_one_track(model_genre, model_name, [0, 100, 200, 300, 400, 500, 600 ,700 ,800, 900], audioSet, audioOptions, 100, frames = 100, Fs = 22050, show_plot = True)
plot_history('genre_full','genre_full_artist_full_key_full', show_plot=True) # artist_full

#######################
# VAE EVALUATION
#######################

#%% Feed the data forward in the CNN
model_name = 'genre_full_artist_full_key_full'
model_base = cnn_load.load_CNN_model(model_name)
X_embed = np.asarray(model_base.predict(X, verbose = 1))

#%% Feed the data forward to the VAE and evaluate
model_vae_name = 'vae_full_beta1_1layer800'
vae_eval.plot_perfs(model_vae_name)
vae_eval.t_sne_cnn_tasks(X, chunks_list, audioSet)

######################
# ALL
######################

# Evaluate the full process using the custom design toymix
# TODO : Missing feature, to be merged soon
# vae_eval.t_sne_toymix(X_embed, model_vae_name, chunks_list, audioSet)
