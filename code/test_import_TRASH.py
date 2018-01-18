import torch
import tensorflow
import keras
import data
from similarity_learning.models.asynchronous.asynchronous import asynchronous_learning
from keras.backend.tensorflow_backend import set_session
#%%

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.visible_device_list = "2"
set_session(tf.Session(config=config))
#%%
audioSet, audioOptions = data.import_data.import_data()
#%%
transform_type, transform_options = audioSet.getTransforms()
audioSet.files = audioSet.files[0:10]
nb_frames = 100
mod_options = {
        'activation': 'relu',
        'batchNormConv': True,
        'FC number': 2048,
        'batchNormDense': True,
        'Alphabet size': 10,
        'Freeze layer': False}
model_name = 'test'
asynchronous_learning(audioSet, audioOptions, nb_frames, mod_options, model_name)
