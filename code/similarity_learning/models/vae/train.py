# -*- coding: utf-8 -*-

""" Training mode.
1. Get the dataflow from the CNN’s latent space (1 chunk = 1 point + 1 label)
2. Build VAE base architecture
3. Train the VAE unsupervised on this data (1 track = 1 valid path)
4. Adjust the parameters to get a better embedding space
5. Freeze the model (save)
"""


# VAE model

# VAE training
# Task : auto-encoding of the chunks' features (sort of t-sne : organise space), dim 10.
# Alternatively, concatenate 3 chunks and organise a "track space"

# Save obtained vae to saved_models

import sys
sys.path.append('similarity_learning/models/vae/')

import VAE
import numpy as np

#%% Fake dataset
input_dim = 1000
nb_chunks = 10000
label_dim = 10
data = np.random.rand(nb_chunks,1000).astype('float32')
labels = np.zeros((nb_chunks, label_dim)).astype('int32')

#%% Make model
model_type="dlgm"
use_label, vae = VAE.build_model(model_type, input_dim, label_dim)

#%% Defining optimizer
trainingOptions = {'lr':1e-4}
vae.init_optimizer(usePyro=False, optimArgs=trainingOptions)

use_cuda = False
if use_cuda:
    vae.cuda()

#%% Define training setting and train
batch_size=100
max_epochs=1 # max_epochs = 1000
vae = VAE.train_vae(vae, data, max_epochs, batch_size, model_type, labels, use_cuda)

#%% Save model once trained
test_name = "test_vae_1"
save_dir = 'similarity_learning/models/vae/saved_models/'
vae.save(save_dir + test_name + '.t7')




