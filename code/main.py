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
from pre_processing.chunkify import track_to_chunks
import skimage.transform as skt
from data.sets.audio import DatasetAudio, importAudioData
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


#%%
file = audioSet.files[0]
downbeat = audioSet.metadata['downbeat'][0][0]
Fs = 44100
chunks = track_to_chunks(0, Fs, downbeat)

data = []
meta = []
data_out = []
#print('loading '+ dataIn[idx
for i in range(len(chunks)):
    chunk = chunks[i].get_cqt(audioSet, options)
    nbBins = chunk.shape[0]
    chunk = skt.resize(chunk, (nbBins, 100), mode='reflect')
    data.append(chunk)
    meta.append(chunks[i].get_meta(audioSet,'genre'))
    data_out.append(model_base.predict(chunk, verbose = 1))


print(len(data_out))
# Feed the data to the VAE

# Re-synthetize data (auto-DJ)
