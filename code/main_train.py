#################################################################
# Here the file to run to execute the full process in train mode
#################################################################

import torch
import tensorflow as tf
import keras
import data
import pickle
import skimage.transform as skt
from keras.backend.tensorflow_backend import set_session
import numpy as np

from similarity_learning.models.asynchronous.asynchronous import asynchronous_learning
import similarity_learning.models.dielemann.load
from pre_processing.chunkify import track_to_chunks
import similarity_learning.models.vae.train as vae_train

#%% GPU configuration: check GPU number for training
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.visible_device_list = "2"
set_session(tf.Session(config=config))


#%% Load data
audioSet, audioOptions = data.import_data.import_data()

#%% Pre-process data

#%% Train CNNs asynchronously + save model

#%%

model_name = 'genre_full'
model_base, _ = similarity_learning.models.dielemann.load.load_CNN_model(model_name)

#%% 
Fs = 22050

nb_files = len(audioSet.files)
# print("~~~~~~~~~~~~~~~~~~~~~~", nb_files)

X = []

for file_id in range(5): # nb_files

    downbeat = audioSet.metadata['downbeat'][file_id][0]
    chunks = track_to_chunks(file_id, Fs, downbeat)
    data = []

    for i in range(len(chunks)):
        chunk = chunks[i].get_cqt(audioSet, audioOptions)
        nbBins = chunk.shape[0]
        chunk = skt.resize(chunk, (nbBins, 100), mode='reflect')
        data.append(chunk)

    # print(np.asarray(data).shape)
    x = np.zeros((len(data), data[0].shape[0], data[0].shape[1]))
    for i in range(len(data)):
        x[i] = data[i]
    x = np.swapaxes(np.array(data),1,2)
    print(x.shape)
    X.append(x)

X = np.asarray(X)
print(X.shape)


#%% Train VAE + save model
# X_embed = np.asarray(model_base.predict(X, verbose = 1))
# print(X_embed.shape)
# vae_train.train_and_save(X_embed)

