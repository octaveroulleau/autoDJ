import torch
import tensorflow as tf
import keras
import data
from similarity_learning.models.asynchronous.asynchronous import asynchronous_learning
from keras.backend.tensorflow_backend import set_session
#%%

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))

#%%
audioSet, audioOptions = data.import_data.import_data()
#%%
transform_type, transform_options = audioSet.getTransforms()
audioSet.files = audioSet.files
batch_size = 20
nb_frames = 4000
mod_options = {
        'activation': 'relu',
        'batchNormConv': True,
        'FC number': 2048,
        'batchNormDense': True,
        'Alphabet size': 10,
        'Freeze layer': False,
        'batch size': batch_size}

model_name = 'key_full'
asynchronous_learning(audioSet, audioOptions, nb_frames, mod_options, model_name, batch_size = batch_size, task = "key")

model_name = 'artist_full'
asynchronous_learning(audioSet, audioOptions, nb_frames, mod_options, model_name, batch_size = batch_size, task = "artist")

model_name = 'genre_full'
asynchronous_learning(audioSet, audioOptions, nb_frames, mod_options, model_name, batch_size = batch_size, task = "genre")

model_name = '_artist_full'
asynchronous_learning(audioSet, audioOptions, nb_frames, mod_options, model_name, batch_size = batch_size, task = "artist", transfer_learning = True, model_base_name = 'genre_full')

model_name = '_key_full'
asynchronous_learning(audioSet, audioOptions, nb_frames, mod_options, model_name, batch_size = batch_size, task = "key", transfer_learning = True, model_base_name = 'genre_full_artist_full')
