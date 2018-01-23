#####################################################################
# Here the file to run to execute the full process in evaluation mode
####################################################################

import numpy as np

import data
import similarity_learning.models.vae.evaluate as vae_eval
import similarity_learning.models.dielemann.load as cnn_load
import pre_processing.load_for_nns as nns_data_load
import pre_processing.chunkify as pr
from re_synthesis.const import SR

# #%% Load audio data and pre-process
# audioSet, audioOptions = data.import_data.import_data()
# chunks_list = pr.dataset_to_chunkList(audioSet, int(SR))
# X = nns_data_load.preprocess_for_cnn(audioSet, audioOptions, 450) # 150 len(audioSet.files)

# #######################
# # CNN EVALUATION
# #######################

# #%% Feed the data forward in the CNN
# model_name = 'genre_full_artist_full_key_full'
# model_base = cnn_load.load_CNN_model(model_name)
# X_embed = np.asarray(model_base.predict(X, verbose = 1))

# Evaluate the model :
# T-SNE
# MIREVAL

#######################
# VAE EVALUATION
#######################

# Evaluate the model :
# T-SNE
# MIREVAL

#%% Feed the data forward to the VAE and evaluate
# model_vae_name = 'vae_genre_full_artist_full_key_full_small'
model_vae_name = 'vae_full_testparams'
vae_eval.plot_perfs(model_vae_name)
# vae_eval.t_sne_cnn_tasks(X, chunks_list, audioSet)

######################
# ALL
######################

# Evaluate the full process using the custom design toymix
# vae_eval.t_sne_toymix(X_embed, model_vae_name, chunks_list, audioSet)
