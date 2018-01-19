###################
# Here the file to run to execute the full process in train mode
###################

import similarity_learning.models.vae.train as vae_train
import numpy as np

#%% Load data

#%% Pre-process data

#%% Train CNNs asynchronously + save model

#%% Train VAE + save model

# Fake dataset
input_dim = 1000
nb_chunks = 10000
data = np.random.rand(nb_chunks,1000).astype('float32')

vae_train.train_and_save(data)