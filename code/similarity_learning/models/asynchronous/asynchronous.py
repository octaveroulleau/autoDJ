# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 18:06:21 2018

@author: octav
"""

from similarity_learning.models.asynchronous.task import AsynchronousTask
from data.sets.audio import DatasetAudio, importAudioData
#from similarity_learning.models.dielemann.build import *
import pdb
import time
import similarity_learning.models.dielemann.build as build
import numpy as np
import keras
import pickle


def asyncTaskPointer(idx, dataIn, options):
    '''
    TO DO
    - Call track to chunk function and pass data and metadata through
    - Fix meta import
    '''
    audioSet = options['audioSet']
    
    audioSet.importMetadataTasks();
    meta = audioSet.metadata['artist'][idx]
 
    print('loading'+ dataIn[idx]+'\nIndex: '+str(idx)+'\nArtist: '+str(meta))
    data, meta_trash = importAudioData(dataIn, options)

    return data, meta
    
def asynchronous_learning(audioSet, audioOptions, nb_frames, model_options, model_name, freq_bins = 168, batch_size = 5, nb_epochs = 5):
    '''
    TO DO:
    -Define in call:  number of frames per chunk, type of model, model options
    -Add number of frames to audioOptions
    -Create model based on model options
    '''
    asyncTask = AsynchronousTask(asyncTaskPointer, numWorkers = 4, batchSize = 5, shuffle = True)
    options = audioOptions
    options["audioSet"] = audioSet  
    alphabet_size = len(set(audioSet.metadata["artist"]))
    
    model_options["Alphabet size"] = alphabet_size
    
    base_model = build.build_conv_layers(nb_frames, freq_bins, model_options)
    full_layer = build.add_fc_layers(base_model, model_options)
    

    for epoch in range(nb_epochs):
        asyncTask.createTask(audioSet.files, options)
        print('Epoch #' + str(epoch));
        for batchIDx, (currentData, currentMeta) in enumerate(asyncTask):
            print('boucle')
            print('[Batch ' + str(batchIDx) + '] Learning step on ' + str(len(currentData)) + ' examples');
            reshape_data(currentData[batchIDx], currentMeta, alphabet_size);
        print('Finished epoch #'+str(epoch))
    
    return 0#model_full, model_base

def reshape_data(currentData, currentMeta, alphabet_size):
    #currentData is size (audioSetSize,batchSize,freq,frames)
    pdb.set_trace()
    train_test = int((len(currentMeta)-1)*0.85)
    x_train = np.swapaxes(np.array(currentData[:train_test]),1,2)
    y_train = np.array(currentMeta[:train_test])
    x_test = np.swapaxes(np.array(currentData[train_test:]),1,2)
    y_test = np.array(currentMeta[train_test:])
    y_train = keras.utils.to_categorical(y_train, alphabet_size)
    y_test = keras.utils.to_categorical(y_test, alphabet_size)
    
    return x_train, y_train, x_test, y_test

def SaveModel(model_full,
              model_base,
              name,
              history,
              pathmodel='../dielemann/models'):
    '''
    Save a model (named with name and the actual date) in the required ./models
    directory.

    Parameters
    ----------
    model: keras.model
        Model to save
    name: string
        Name of the model
    pathmodel (optionnal): string
        Path to the models repository

    Returns
    -------
    model: keras.model
        The model we saved.
    '''
    print('Save model to ...')
    date = time.ctime()
    date = date.replace(' ', '_')
    date = date.replace(':', '-')
    print('Save base model as ' + name + '_base_' + date + '.h5' + '...')
    filepath_base = pathmodel + name + '_base_' + date + '.h5'
    model_base.save(filepath_base)
    print('Base model saved in '+ filepath_base)

    print('Save full model as ' + name + '_full_' + date + '.h5' + '...')
    filepath_full = pathmodel + name + '_full_' + date + '.h5'
    model_full.save(filepath_full)
    print('Base model saved in '+ filepath_full)
    
    print('Save history as ' + name + '_history_' + date + '...')
    file = open('../dielemann/models/' + name + '_history_' + date)
    pickle.dump(history.history, file)
    print('History saved in' + file)
    
    
    
    
