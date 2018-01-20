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

""" 
#######################
# BEATS EVALUATION
#######################

# load reference data
beats_ref = load_beats_reference()

# compute beats with several algorithms
beats_madmom = compute_beats_madmom()
beats_librosa = compute_beats_librosa()
beats_ellis = compute_beats_ellis()

# calculate score
beats_score_madmom = mir_eval_beats(beats_ref, beats_madmom)
beats_score_librosa = mir_eval_beats(beats_ref, beats_librosa)
beats_score_ellis = mir_eval_beats(beats_ref, beats_ellis)

#########################
# DOWNBEATS EVALUATION
#########################

# load reference data
downbeats_ref = load_downbeats_reference()

# compute downbeats with several algorithms
downbeats_madmom = compute_downbeats_madmom()

# calculate score
downbeats_score_madmom = mir_eval_downbeats(downbeats_ref, downbeats_madmom)
"""

#######################
# CNN EVALUATION
#######################

# Feed the data forward in the CNN

# import matplotlib.pyplot as plt
# import pickle
# tsne = pickle.load(open('./tsne', 'rb'))
# #%%

# tsne = tsne[1:]

# plt.figure()
# for i in range(len(tsne)):
#     for j in range(len(tsne[i])):
#         plt.scatter(tsne[i][j][0][0], tsne[i][j][0][1] )

# plt.show()

# Evaluate the model :
# T-SNE
# MIREVAL

#######################
# VAE EVALUATION
#######################

# # Fake dataset
# input_dim = 1000
# nb_chunks = 123
# data = np.random.rand(nb_chunks,input_dim).astype('float32')
# # Feed to the VAE and return indexes of nearest chunks
# idx_nearest_chunks = vae_eval.evaluate(data)
# # Evaluate the model :
# # T-SNE
# # MIREVAL


#%% Load audio data and pre-process
audioSet, audioOptions = data.import_data.import_data()
chunks_list = pr.dataset_to_chunkList(audioSet, int(SR))
X = nns_data_load.preprocess_for_cnn(audioSet, audioOptions, 50)

#%% Feed the data forward in the CNN
model_name = 'genre_full'
model_base, _ = cnn_load.load_CNN_model(model_name)
X_embed = np.asarray(model_base.predict(X, verbose = 1))

#%% Feed the data to the VAE
vae_eval.evaluate(X_embed)

######################
# ALL
######################

# Toymix here