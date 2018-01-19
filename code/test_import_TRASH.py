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
model_name = 'artist_full'
asynchronous_learning(audioSet, audioOptions, nb_frames, mod_options, model_name, batch_size = batch_size, task = "artist")
#%%
'''
import pickle
import matplotlib.pyplot as plt
import numpy
#%%
history = pickle.load(open('./similarity_learning/models/dielemann/models/genre_full_history','rb'))
history_epoch = {}
for epoch in range(200):
    batch_number = 0
    history_epoch['epoch '+str(epoch)] = {}
    history_epoch['epoch '+str(epoch)]['acc'] = 0
    history_epoch['epoch '+str(epoch)]['loss'] = 0
    history_epoch['epoch '+str(epoch)]['val_acc'] = 0
    history_epoch['epoch '+str(epoch)]['val_loss'] = 0
    for batch in range(int(np.floor(1000/20))):
        if "epoch "+str(epoch)+" batch "+str(batch) in history:
            batch_number = batch_number + 1
            history_epoch['epoch '+str(epoch)]['acc'] = history_epoch['epoch '+str(epoch)]['acc'] + history["epoch "+str(epoch)+" batch "+str(batch)]['acc'][0]
            history_epoch['epoch '+str(epoch)]['loss'] = history_epoch['epoch '+str(epoch)]['loss'] + history["epoch "+str(epoch)+" batch "+str(batch)]['loss'][0]
            history_epoch['epoch '+str(epoch)]['val_acc'] = history_epoch['epoch '+str(epoch)]['val_acc'] + history["epoch "+str(epoch)+" batch "+str(batch)]['val_acc'][0]
            history_epoch['epoch '+str(epoch)]['val_loss'] = history_epoch['epoch '+str(epoch)]['val_loss'] + history["epoch "+str(epoch)+" batch "+str(batch)]['val_loss'][0]
    history_epoch['epoch '+str(epoch)]['acc'] = history_epoch['epoch '+str(epoch)]['acc']/batch_number
    history_epoch['epoch '+str(epoch)]['loss'] = history_epoch['epoch '+str(epoch)]['loss']/batch_number
    history_epoch['epoch '+str(epoch)]['val_acc'] = history_epoch['epoch '+str(epoch)]['val_acc']/batch_number
    history_epoch['epoch '+str(epoch)]['val_loss'] = history_epoch['epoch '+str(epoch)]['val_loss']/batch_number
#%%
val_acc = []
val_loss = []
acc = []
loss = []

for i in history_epoch:
    val_loss.append(history_epoch[i]['val_loss'])
    val_acc.append(history_epoch[i]['val_acc'])
    acc.append(history_epoch[i]['acc'])
    loss.append(history_epoch[i]['loss'])

#%%
plt.figure()
plt.plot(range(200), val_loss)
plt.figure()
plt.plot(range(200), val_acc)
'''
