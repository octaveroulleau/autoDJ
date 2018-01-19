###################
# Here the file to run to execute the full process in normal (forward) mode
###################

import similarity_learning.models.vae.compose as vae_comp
import numpy as np

#%% Load the data and pre-process them
chunk_list = your_function() # only metadata, no need for audio or cqt here

#%% Feed the data forward in the CNN

#%% Feed the data to the VAE

# Fake dataset
input_dim = 1000
nb_chunks = 123
data = np.random.rand(nb_chunks,input_dim).astype('float32')

# Feed to the VAE and return indexes of nearest chunks
idx_nearest_chunks = vae_comp.compose_line(data)
# From this, establish the list of mixing points
mp_list = chunks_to_mp(idx_nearest_chunks, chunks_list)

#%% Re-synthetize data (auto-DJ)
finalset, sr = re.compose_track(mp_list)
re.write_track(np.array(finalset),sr, 'test.wav')