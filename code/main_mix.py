##################################################################
# Here the file to run to execute the full process in forward mode
##################################################################

import numpy as np
import similarity_learning.models.vae.compose as vae_comp
import pre_processing.chunkify as pr
import re_synthesis.assemble_mixing_points as re

#%% Load the data and pre-process them
# Load the metadata
audioSet = pr.load_dataset()
chunks_list = pr.dataset_to_chunkList(audioSet, 22050)
# Load the audio data and pre-process

#%% Feed the data forward in the CNN

#%% Feed the data to the VAE
# Fake dataset
input_dim = 1000
nb_chunks = 123
data = np.random.rand(nb_chunks,input_dim).astype('float32')
# Feed to the VAE and return indexes of nearest chunks
idx_nearest_chunks = vae_comp.compose_line(data)
# print(idx_nearest_chunks)
# From this, establish the list of mixing points
mp_list = vae_comp.chunks_to_mp(idx_nearest_chunks, chunks_list, audioSet)
print(mp_list)

#%% Re-synthetize data (auto-DJ)
finalset, sr = re.compose_track(mp_list)
re.write_track(np.array(finalset),sr, 'test.wav')