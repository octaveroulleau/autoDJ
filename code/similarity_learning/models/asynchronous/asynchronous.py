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
from pre_processing.chunkify import track_to_chunks



def asyncTaskPointer(idx, dataIn, options):
    '''
    TO DO
    - Call track to chunk function and pass data and metadata through
    - Fix meta import
    '''
    audioSet = options['audioSet']
    
    downbeat = audioSet.metadata["downbeat"][idx][0]
    Fs = 44100
    chunks = track_to_chunks(idx, Fs, downbeat)
    
    data = []
    meta = []
    print('loading '+ dataIn[idx])

    for i in range(len(chunks)):
        print(chunks[i])
        chunk = chunks[i].get_cqt(audioSet, options, target_frames = 100)
        print('Ok')
        data.append(chunk)
        meta.append(chunks[i].get_meta(audioSet,options['task']))
        
    print(str(len(data)) + ' chunks created')

        
    
    """
    audioSet = options['audioSet']
    
    audioSet.importMetadataTasks();
    meta = audioSet.metadata[options["task"]][idx]
 
    print('loading '+ dataIn[idx])
    data, meta_trash = importAudioData(dataIn, options)
    
    #chunks, chunks_meta = Chunks
    """

    return data, meta
    
def asynchronous_learning(audioSet, audioOptions, nb_frames, model_options, model_name, task = "genre", freq_bins = 168, batch_size = 5, nb_epochs = 5):
    '''
    TO DO:
    -Define in call:  number of frames per chunk, type of model, model options
    -Add number of frames to audioOptions
    -Create model based on model options
    '''
    asyncTask = AsynchronousTask(asyncTaskPointer, numWorkers = 4, batchSize = 5, shuffle = True)
    options = audioOptions
    options["audioSet"] = audioSet  
    options["task"] = task
    options["frames number"] = nb_frames
    alphabet_size = len(set(audioSet.metadata[task]))
    
    model_options["Alphabet size"] = alphabet_size
    
    model_base = build.build_conv_layers(nb_frames, freq_bins, model_options)
    model_full = build.add_fc_layers(model_base, model_options)
    

    for epoch in range(nb_epochs):
        asyncTask.createTask(audioSet.files, options)
        print('Epoch #' + str(epoch));
        for batchIDx, (currentData, currentMeta) in enumerate(asyncTask):
            print('boucle')
            print('[Batch ' + str(batchIDx) + '] Learning step on ' + str(len(currentData[batchIDx])) + ' examples');
            x_train, x_test = reshape_data(currentData[batchIDx], currentMeta, alphabet_size);
            #history = model_full.fit(x_train, y_train, batch_size = batchSize, epochs = 1, verbose = 1, validation_split = 0.2)
        print('Finished epoch #'+str(epoch))
    
    #save_model(model_full, model_base, history, model_name)
    return 0#model_full, model_base

def reshape_data(currentData, currentMeta, alphabet_size):
    #currentData is size (audioSetSize,batchSize,freq,frames)
    pdb.set_trace()
    x_train = np.swapaxes(np.array(currentData),1,2)
    y_train = np.array(currentMeta)
    y_train = keras.utils.to_categorical(y_train, alphabet_size)
    return x_train, y_train

def save_model(model_full,
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
    
    
    
    
