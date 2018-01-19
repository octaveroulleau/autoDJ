# Here the file to run to execute the full process in normal (forward) mode

# Load the data and pre-process them

# Feed the data forward in the CNN

#%%
import torch
import tensorflow as tf
import keras
import data
import similarity_learning.models.dielemann.load 
from keras.backend.tensorflow_backend import set_session
import pdb
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
audioSet.files = audioSet.files
batch_size = 20
nb_frames = 100
mod_options = {
        'activation': 'relu',
        'batchNormConv': True,
        'FC number': 2048,
        'batchNormDense': True,
        'Alphabet size': 10,
        'Freeze layer': False,
        'batch size': batch_size}
model_name = 'genre_full'
model_base, model_options = similarity_learning.models.dielemann.load.load_CNN_model(model_name)
print(model_base)
print(len(model_options))
# Feed the data to the VAE

# Re-synthetize data (auto-DJ)
