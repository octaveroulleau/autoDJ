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
    meta = [idx, meta]
    data, meta_trash = importAudioData(dataIn, options)

    return data, meta
    
def asynchronous_learning(audioSet, audioOptions, nb_frames, model_options, batch_size = 5, nb_epochs = 5):
    '''
    TO DO:
    -Define in call:  number of frames per chunk, type of model, model options
    -Add number of frames to audioOptions
    -Create model based on model options
    '''
    asyncTask = AsynchronousTask(asyncTaskPointer, numWorkers = 4, batchSize = 5, shuffle = True)
    options = audioOptions
    options["audioSet"] = audioSet
    
    pdb.set_trace()
    data_init, meta = importAudioData(audioSet.files[0], options)
    
    
    #base_model = build.build_conv_layers(nb_frames, )
    

    for epoch in range(nb_epochs):
        asyncTask.createTask(audioSet.files, options)
        print('Epoch #' + str(epoch));
        for batchIDx, (currentData, currentMeta) in enumerate(asyncTask):
            print('boucle')
            print('[Batch ' + str(batchIDx) + '] Learning step on ' + str(len(currentData)) + ' examples');
            dummy_learn(currentData, currentMeta);
        print('Finished epoch #'+str(epoch))
    
    return 0#model_full, model_base

def dummy_learn(currentData, currentMeta):
    #currentData is size (audioSetSize,batchSize,freq,frames)
    print('Learning on current data - size :');
    print(len(currentData))
    # Simulate time
    print('Learning on ID #' + str(currentMeta[0])) #+ " - 1st elt : " + str(currentData[t][1][1]))
    print(currentMeta[1])
