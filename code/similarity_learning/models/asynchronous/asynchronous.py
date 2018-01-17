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

def asyncTaskPointer(idx, dataIn, options):
    pdb.set_trace()
    '''
    TO DO
    - Call track to chunk function and pass data and metadata through
    - Fix meta import
    '''
    print('loading'+ dataIn[idx])
    
    data, meta = importAudioData(dataIn, options)
    meta = options['metadata']['artist'][idx]
    return data, meta
    
def asynchronous_learning(audioSet, audioOptions, batch_size = 5, nb_epochs = 5):
    '''
    TO DO:
    -Define in call:  number of frames per chunk, type of model, model options
    -Add number of frames to audioOptions
    -Create model based on model options
    '''
    asyncTask = AsynchronousTask(asyncTaskPointer, numWorkers = 4, batchSize = 5, shuffle = False)
    options = audioOptions
    options.update({"metadata":audioSet.metadata})
    asyncTask.createTask(audioSet.files, options)

    for epoch in range(nb_epochs):
        print('Epoch #' + str(epoch));
        for batchIDx, (currentData, currentMeta) in enumerate(asyncTask):
            print('boucle')
            print('[Batch ' + str(batchIDx) + '] Learning step on ' + str(len(currentData)) + ' examples');
            dummy_learn(currentData, currentMeta);
        print('Finished epoch #'+str(epoch))
    
    return 0#model_full, model_base

def dummy_learn(currentData, currentMeta):
    #currentData is size (audioSetSize,batchSize,freq,frames)
    pdb.set_trace()
    print('Learning on current data - size :');
    print(len(currentData))
    # Simulate time
    for t in range(len(currentData)):
        print('Learning on ID #' + str(currentMeta[0])) #+ " - 1st elt : " + str(currentData[t][1][1]))

