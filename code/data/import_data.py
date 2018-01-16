# -*- coding: utf-8 -*-
"""

 This file serves as a base script for the PAM Auto-DJ project
    * Import GTZAN dataset
    * Import metadata for various tasks
        - Artist
        - Beat
        - Downbeat
        - Genre
        - Key
        - Tempo
    * Import different types of spectral transports
        - Mel-spectrogram
        - Constant-Q transform
        - Modulation spectrum
    * Plot data and metadata on top of some tracks
    * Perform some pre-processing on the data
    * Window the input on a given number of beats
    * Show asynchronous learning on tracks

 Author : Philippe Esling
          <esling@ircam.fr>

 Version : 0.1

"""
import pdb
# Import our classes
from data.sets.audio import DatasetAudio
#from asynchronous.task import AsynchronousTask
# External libs import
#from matplotlib import pyplot as plt
import skimage.transform as skt
import numpy as np
#import dielemann.build
#import keras
#from data_to_phrase import *


def import_data(pamTransforms = 'cqt'):
    #%%
    # Root to find the GTZAN dataset
    baseFolder = "../../../autodj_sets/datasets"
    # All metadatas that we will import
    taskList = ["artist", "beat", "downbeat", "genre", "key", "tempo"]
    # All transforms that we should import
    #pamTransforms = ["cqt","mel"]

    """
    ###################################
    # [Audio import]
    ###################################
    """

    audioOptions = {
    "dataDirectory":baseFolder+'/gtzan/data',
    "metadataDirectory":baseFolder+'/gtzan/metadata',
    "dataPrefix":baseFolder,
    "analysisDirectory":baseFolder+'/gtzan/transforms/',    # Root to place (and find) the transformed data
    "importType":'asynchronous',                            # Type of import (direct or asynchronous)
    "transformType":['stft'],                               # Type of transform (can be a list)
    "tasks":taskList,                                       # Tasks to import
    "verbose":True,                                         # Be verbose or not
    "checkIntegrity":False,                                 # Check that files exist (while loading)
    "forceRecompute":False                                  # Force the update
    };

    # Create dataset
    audioSet = DatasetAudio(audioOptions);
    #%%
    ##################u
    # Metadata import operations
    ##################
    print('[Import metadata]');
    audioSet.importMetadataTasks();
    #%% Print the contents of metadata
    for task in taskList:
        print("  * Checking " + task);
        #    print(audioSet.metadata[task]);
        metadata = audioSet.metadata[task];
        print('\t Number of annotated \t : ' + str(len(metadata)))
        # Check tempo task
        if (task == 'tempo'):
            print('\t Tempo values :')
            metadata = np.array(metadata);
            print('\t\t Min \t : %f\n\t\t Max \t : %f' % (np.min(metadata), np.max(metadata)))
            print('\t\t Mean \t : %f\n\t\t Var \t : %f\n' % (np.mean(metadata), np.std(metadata)))
        if (task == 'beat' or task == 'downbeat'):
            print('\t Number of annotations per track :')
            finalStats = np.zeros(len(metadata))
            for vals in range(len(metadata)):
                finalStats[vals] = metadata[vals][0].size
            print('\t\t Min \t : %f\n\t\t Max \t : %f' % (np.min(finalStats), np.max(finalStats)))
            print('\t\t Mean \t : %f\n\t\t Var \t : %f\n' % (np.mean(finalStats), np.std(finalStats)))
        else:
            if (audioSet.classes[task]) and (audioSet.classes[task]["_length"] > 0):
                print('\t Number of classes \t : ' + str(audioSet.classes[task]["_length"]))
                print('\t Number of instances per class :')
                tmpClasses = np.zeros(audioSet.classes[task]["_length"])
                metadata = np.array(metadata)
                curID = 0
                for k, v in audioSet.classes[task].items():
                    if (k != "_length"):
                        tmpClasses[curID] = np.sum((metadata == v) * 1)
                        curID = curID + 1
                print('\t\t Min \t : %f\n\t\t Max \t : %f' % (np.min(tmpClasses), np.max(tmpClasses)))
                print('\t\t Mean \t : %f\n\t\t Var \t : %f\n' % (np.mean(tmpClasses), np.std(tmpClasses)))
    #%%
    ##################
    # Data import operations
    ##################
    # Prepare asynchronous pointers
    transformList, transformOptions = audioSet.getTransforms();
    tempAnalysis = audioSet.analysisDirectory
    audioSet.transformType = [pamTransforms]
    audioSet.analysisDirectory = tempAnalysis + pamTransforms + '/'

    return audioSet, audioOptions

'''
#%%
measures_dict = []
genre_dict = []
for transform in pamTransforms[:1]:

    print('[Import data for ' + transform + ']');
    # Import spectral data
    audioSet.transformType = [transform];
    audioSet.analysisDirectory = tempAnalysis + transform + '/'
    audioSet.importData(None, {"transformOptions":transformOptions});
    #%%
    print('[Plot data for ' + transform + ']');

    measures_dict, meta_dict = data_to_phrase(audioSet, 4, 4000)
'''
'''
    # Plot data and metadata on top of some tracks
    idList = [0, 101, 202, 303, 404, 505, 606, 707, 808, 909];
    for curTrack in idList:
        curData = audioSet.data[curTrack];
        nbBins, nbFrames = curData.shape[0], curData.shape[1]
        # Retrieve metadata for the current track
        curGenre = audioSet.metadata["genre"][curTrack];
        curArtist = audioSet.metadata["artist"][curTrack];
        curDownbeats = audioSet.metadata["downbeat"][curTrack][0];
        curBeats = audioSet.metadata["beat"][curTrack][0];
        curTempo = audioSet.metadata["tempo"][curTrack];
        curKey = audioSet.metadata["key"][curTrack];
        # Find the strings for some metadata
        genreStr = list(audioSet.classes["genre"].keys())[curGenre+1];
        artistStr = list(audioSet.classes["artist"].keys())[curArtist+1];
        keyStr = list(audioSet.classes["key"].keys())[curKey+1];
        Plot stuffs

        plt.figure(figsize=(17, 7))
        plt.imshow(np.flipud(curData))
        plt.title('Track n' + str(curTrack) + ' - ' + artistStr + ' - ' + genreStr + ' - ' + keyStr)
        # Now plot the downbeats information
        trackLenS = 30
        pointRatio = nbFrames / trackLenS
        for d in curDownbeats:
            fIDx = d * pointRatio
            plt.plot([fIDx, fIDx], [1, nbBins-1], 'r-', lw=4)
        # Do the same with beats (less thick)
        for d in curBeats:
            fIDx = d * pointRatio
            plt.plot([fIDx, fIDx], [1, nbBins-1], 'k-', lw=1)
        plt.axis('tight')
        # Now extract the first three windows to plot
        plt.figure(figsize=(11,7))

        for t in range(3):
            if (t < len(curDownbeats) - 1):
                sFrame, eFrame = (curDownbeats[t] * pointRatio), (curDownbeats[t+1] * pointRatio)
                curMeasure = curData[:, int(sFrame):int(eFrame)]
                # Perform local zero-mean unit-var
                curMeasure = (curMeasure - np.mean(curMeasure)) / np.max(curMeasure)
                # Here show how we can perform resampling on these windows
                curMeasure = skt.resize(curMeasure, (nbBins, 500), mode='reflect')
                # This allows to have all sub-tracks in a tempo-invariant style !
                plt.subplot(1, 3, t+1)
                plt.imshow(np.flipud(curMeasure))
                plt.axis('tight')
        t = 0
        trackLenS = 30
        pointRatio = nbFrames / trackLenS
        #Take the data from each 16 beats
        while t < len(curDownbeats)-4:
            sFrame, eFrame = (curDownbeats[t]*pointRatio), (curDownbeats[t+4]*pointRatio)
            curPhrase = curData[:, int(sFrame):int(eFrame)]
            #Perform local zero-mean unit-var
            curPhrase = (curPhrase - np.mean(curPhrase)) / np.max(curPhrase)
            # Here show how we can perform resampling on these windows
            curPhrase = skt.resize(curPhrase, (nbBins, 4000), mode='reflect')
            measures_dict.append(curPhrase)#, curGenre, genreStr])
            genre_dict.append(curGenre)
            t += 4

        if not t==len(curDownbeats)-1:
            sFrame, eFrame = (curDownbeats[t]*pointRatio), (curDownbeats[len(curDownbeats)-1]*pointRatio)
            curPhrase = curData[:, int(sFrame):int(eFrame)]
            #Perform local zero-mean unit-var
            curPhrase = (curPhrase - np.mean(curPhrase))/ np.max(curPhrase)
            # Resampling the phrases: here, each phrase will contain 4000 samples
            curPhrase = skt.resize(curPhrase, (nbBins, 4000), mode = 'reflect')
            measures_dict.append(curPhrase)#, curGenre, genreStr])
            genre_dict.append(curGenre)
           '''
'''
# Perform some pre-processing on the data
#%%
alphabet = np.array(list(set(genre_dict)))
alphabet_size = len(alphabet)
train_test = int((len(measures_dict)-1)*0.85)
x_train = np.swapaxes(np.array(measures_dict[:train_test]),1,2)
y_train = np.array(genre_dict[:train_test])
x_test = np.swapaxes(np.array(measures_dict[train_test:]),1,2)
y_test = np.array(genre_dict[train_test:])
#%%

frames = np.shape(x_train)[1]
freq_bins = np.shape(x_train)[2]
y_train = keras.utils.to_categorical(y_train, alphabet_size)
y_test = keras.utils.to_categorical(y_test, alphabet_size)
#%%

mod_options = {
        'activation': 'relu',
        'batchNormConv': True,
        'FC number': 2048,
        'batchNormDense': True,
        'Alphabet size': alphabet_size}
'''
'''
model = dielemann.build.build_full_model(frames, freq_bins, mod_options)
#%%d
model.fit(x_train,y_train,batch_size = 1, epochs = 2, verbose = 1, validation_data = (x_test, y_test))
# Window the input on a given number of beats

# Show asynchronous learning on tracks
'''
