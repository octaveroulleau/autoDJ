import torch
import tensorflow
import keras
import data
from similarity_learning.models.asynchronous.asynchronous import asynchronous_learning
#%%
audioSet, audioOptions = data.import_data.import_data()
#%%
transform_type, transform_options = audioSet.getTransforms()
audioSet.files = audioSet.files[0:300]
nb_frames = 4000
mod_options = {
        'activation': 'relu',
        'batchNormConv': True,
        'FC number': 2048,
        'batchNormDense': True,
        'Alphabet size': 10,
        'Freeze layer': False}
model_name = 'test'
asynchronous_learning(audioSet, audioOptions, nb_frames, mod_options, model_name)
