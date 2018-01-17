"""
#
# This file explains the use of asynchronous tasks (multi-threading jobs)
# in the Aciditeam toolbox prepared for Torch
#
# Asynchronous jobs allow you to get full multi-threaded capabilites
# in order to perform learning, even though you have massive datasets
# to process, transform, import or augment
#
# Author : Philippe Esling
#          <esling@ircam.fr>
#
# Version : 0.9
#
# This code is under LGPL License.
#
#
"""

# Import all corresponding classes
from asynchronous.task import AsynchronousTask
# Other imports for demonstration
import numpy as np
import time

"""
###################################
# First you should understand that asynchronous task are useful
# only if the operation you are trying to multithread requires at least
# a quite large amount of time. Otherwise the overhead is too heavy.
#
# If you see an operation that is quite cumbersome, you should
# create a function that has the following signature
# dataIn = asyncTaskPointer(dataOut, options)
#   - idx            = index of the element inside the data in to handle
#   - dataIn         = input tensor / table containing a _list_ of processing to do
#   - options        = set of auxiliary options, which at contains
#      * Anything passed when using asynchronousTask:createTask()
#   - finalData      = output tensor containing the resulting data
#   - finalLabels    = output tensor containing the resulting labels

"""
# This would be the heavy function to be asynchronous
def asyncMinitaskPointer(idx, dataIn, options):
    finalData = np.zeros((1000, 1000))
    finalLabels = np.zeros(1)
    # Iterate over the dataIn data
    print('Processing ' + dataIn[idx]);
    # Simulate (smaller) processing time
    time.sleep(0.2)
    finalData = finalData + idx
    finalLabels[0] = idx
    # Return the data
    return finalData, finalLabels;

# Here we just make a simulated learning function
def dummyLearn(currentData, currentMeta):
    print('Learning on current data - size :');
    print(len(currentData))
    # Simulate time
    for t in range(len(currentData)):
        print('Learning on ID #' + str(currentMeta[t]) + " - 1st elt : " + str(currentData[t][1][1]))
        for t2 in range(100):
            currentData[0] = currentData[0] + currentData[t];

"""
###################################
# To create your asynchronous task, construct an object with
# asynchronousTask(funcPointer, nbJobs, maxSize, taskName, debugMode)
#   - functionPointer  = computation function
#   - nbJobs           = number of threads to use
#   - maxSize          = maximum number of items (depending on RAM memory or HDD size)
#   - taskName         = name of the task (optional)
#   - debugMode        = set debug mode (optional)
"""
asyncTask = AsynchronousTask(asyncMinitaskPointer, numWorkers=4, batchSize=64, shuffle=False)

"""
###################################
# Once the task is setup, you have only 4 operations to perform
#   - createTask       = setup the current task (data and options)
#   - retrieveData     = get any data currently available
#   - cleanTask        = clean anything left in the task
#   - isFinished       = check if we have finished the task
"""
# Fake dataset for a list of files
datasetFiles = [None] * 300
for f in range(300):
    datasetFiles[f] = 'file_' + str(f) + '.huge'
# No options here
options = {};
# Simulate the learning process
for epoch in range(2):
    print('Epoch #' + str(epoch));
    # Create the task (here simulate a loader)
    asyncTask.createTask(datasetFiles, options);
    # Check if we have processed the whole dataset
    for batchIDx, (currentData, currentMeta) in enumerate(asyncTask):
        print('[Batch ' + str(batchIDx) + '] Learning step on ' + str(len(currentData)) + ' examples');
        # Perform training on current data
        dummyLearn(currentData, currentMeta);
    print('Finished epoch #' + str(epoch))
    # Clean the task
    # asyncTask.cleanTask()

"""
###################################
# The algorithm should also work for inputs that are Tensors
# (Typically a data augmentation job that we simulate here)
# (We also examplify how to pass options to your asynchronous function)
"""
def asyncAugmentationPointer(idx, dataIn, options):
    finalSize = options.augmentationFactor
    finalData = np.zeros(finalSize, 1000, 1000)
    finalLabels = np.zeros(finalSize)
    # Iterate over the dataIn data
    for i in range(finalData.shape[0]):
        print('Augmenting ex.' + str(i))
        # Simulate (smaller) processing time
        time.sleep(0.2)
        # Store the computation
        finalData[i] = np.zeros(1000, 1000) + i;
        finalLabels[i] = i;
    # Return the data
    return finalData, finalLabels;

# Create a augmentation task
augmentTask = AsynchronousTask(asyncAugmentationPointer, numWorkers=4, batchSize=64, shuffle=True)

"""
###################################
# Now what is cool with this formalism is that we can actually create
# multiple asynchronous tasks that depend on each other.
# This means an async task inside an async task
# Here we will try to do a import / augmentation duo
# (Based on the two previous function pointers)
"""
# Fake dataset for a list of files
datasetFiles = [None] * 300
for f in range(300):
    datasetFiles[f] = 'file_' + str(f) + '.huge'
# No options here
options = {};
# Simulate the learning process
for epoch in range(2):
    print('Epoch #' + str(epoch))
    # Create the task (here simulate a loader)
    asyncTask.createTask(datasetFiles, options)
    # Check if we have processed the whole dataset
    for batchIDx, (currentData, currentMeta) in enumerate(asyncTask):
        # Here we transfor the data into a tensor (if needed)
        currentTensor = np.zeros((currentData.shape[0], currentData[0].shape[0], currentData[0].shape[1]))
        for t in range(currentData.shape[0]):
            currentTensor[t] = currentData[t]
        print('Augmenting step on ' + str(len(currentData)) + ' examples');
        # Now create a multi-threaded sub-task for augmentations
        augmentTask.createTask(currentTensor, {"augmentationFactor":4})
        for batchAugIDx, (augmentData, augmentMeta) in enumerate(asyncTask):
            print('Learning step on ' + len(augmentData) + ' examples');
            # Perform training on current data
            dummyLearn(augmentData, augmentMeta);
        # Clean the task
        # augmentTask.cleanTask();
    print('Finished epoch #' + str(epoch))
    # Clean the task
    # asyncTask.cleanTask();
