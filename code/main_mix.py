##################################################################
# Here the file to run to execute the full process in forward mode
##################################################################

import numpy as np

import data
import similarity_learning.models.vae.compose as vae_comp
import pre_processing.chunkify as pr
import re_synthesis.assemble_mixing_points as re
from re_synthesis.const import SR

#%% Load the data and pre-process them
# Load the metadata
audioSet, audioOptions = data.import_data.import_data()
chunks_list = pr.dataset_to_chunkList(audioSet, int(SR))
# Load the audio data and pre-process
X = nns_data_load.preprocess_for_cnn(audioSet, audioOptions, 50)

#%% Feed the data forward in the CNN
model_name = 'genre_full_artist_full_key_full'
model_base = cnn_load.load_CNN_model(model_name)
X_embed = np.asarray(model_base.predict(X, verbose = 1))

#%% Feed the data to the VAE
# Fake dataset
# input_dim = 1000
# nb_chunks = 123
# data = np.random.rand(nb_chunks,input_dim).astype('float32')
# Feed to the VAE and return indexes of nearest chunks
idx_nearest_chunks = vae_comp.compose_line(X_embed)
# print(idx_nearest_chunks)
# From this, establish the list of mixing points
mp_list = vae_comp.chunks_to_mp(idx_nearest_chunks, chunks_list, audioSet)
print(mp_list)

#%% Re-synthetize data (auto-DJ)
finalset = re.compose_track(mp_list)
re.write_track(finalset, 'test.wav')