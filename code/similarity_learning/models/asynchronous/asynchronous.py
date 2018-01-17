# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 18:06:21 2018

@author: octav
"""

from similarity_learning.models.asynchronous.task import AsynchronousTask
from data.sets.audio import DatasetAudio, importAudioData
import pdb
import time

def asyncTaskPointer(idx, dataIn, options):
    print('loading'+ dataIn[idx])
    
    data, meta = importAudioData(dataIn, options)
    return data, meta
    
def asynchronous_learning(audioSet, audioOptions, batch_size = 5, nb_epochs = 5):
    asyncTask = AsynchronousTask(asyncTaskPointer, numWorkers = 4, batchSize = 5, shuffle = False)
    asyncTask.createTask(audioSet.files, audioOptions)
    for batchIDx, (currentData, currentMeta) in enumerate(asyncTask):
        print('boucle')
        print('[Batch ' + str(batchIDx) + '] Learning step on ' + str(len(currentData)) + ' examples');
        dummy_learn(currentData, currentMeta);
    
    return 0

def dummy_learn(currentData, currentMeta):
    print('Learning on current data - size :');
    print(len(currentData))
    # Simulate time
    for t in range(len(currentData)):
        print('Learning on ID #' + str(currentMeta[t]) + " - 1st elt : " + str(currentData[t][1][1]))
        for t2 in range(100):
            currentData[0] = currentData[0] + currentData[t];
    
