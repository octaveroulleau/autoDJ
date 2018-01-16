# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 18:06:21 2018

@author: octav
"""

from similarity_learning.models.asynchronous.task import AsynchronousTask
from data.sets.audio import DatasetAudio, importAudioData
import pdb

def asyncTaskPointer(idx, dataIn, options):
    print('loading'+ dataIn[idx])
    
    data, meta = importAudioData(dataIn, options)
    return data, meta
    
def asynchronous_learning(audioSet, batch_size = 64, nb_epochs = 5):
    asyncTask = AsynchronousTask(asyncTaskPointer, numWorkers = 4, batchSize = 64, shuffle = False)
    transfi_list, tranform_options  = audioSet.getTransforms()
    asyncTask.createTask(audioSet.files, options = {"transformOptions": transform_options})
    for batchIDx, (currentData, currentMeta) in enumerate(asyncTask):
        print('[Batch ' + str(batchIDx) + '] Learning step on ' + str(len(currentData)) + ' examples');
        dummy_learn(currentData, currentMeta);
    
    return 0

def dummy_learn(currentData, currentMeta):
    pdb.set_trace()
    print(currentData)
    print(currentMeta)
    
