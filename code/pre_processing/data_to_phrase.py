# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 16:38:06 2017

@author: octav
"""
import numpy as np
import skimage.transform as skt
import pdb

def data_to_phrase(audioSet,curTrack, nbMeasure, resample_size, meta_value = 'genre'):
    measures_dict = []
    meta_dict = []
    #for curTrack in range(len(audioSet.data)):
    curData = audioSet.data[curTrack];
    nbBins, nbFrames = curData.shape[0], curData.shape[1]
    # Retrieve metadata for the current track
    curMeta = audioSet.metadata[meta_value][curTrack];
    curDownbeats = audioSet.metadata["downbeat"][curTrack][0];
    curBeats = audioSet.metadata["beat"][curTrack][0];
    curTempo = audioSet.metadata["tempo"][curTrack];
    # Find the strings for some metadata
    metaStr = list(audioSet.classes[meta_value].keys())[curMeta+1];

    '''
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
                    '''
    print(curTrack)
    trackLenS = 30
    t = 0   
    pointRatio = nbFrames / trackLenS
    #Take the data from each 16 beats
    if len(curDownbeats) != 0:
        while t < len(curDownbeats)-nbMeasure:
            sFrame, eFrame = (curDownbeats[t]*pointRatio), (curDownbeats[t+nbMeasure]*pointRatio)
            curPhrase = curData[:, int(sFrame):int(eFrame)]
            #Perform local zero-mean unit-var
            curPhrase = (curPhrase - np.mean(curPhrase))/ np.max(curPhrase)
            # Resampling the phrases: here, each phrase will contain 4000 samples
            curPhrase = skt.resize(curPhrase, (nbBins, resample_size), mode = 'reflect')
            measures_dict.append(curPhrase)#, curGenre, genreStr])
            meta_dict.append(curMeta)
            t += 4
    
    if not t==len(curDownbeats)-1:
        sFrame, eFrame = (curDownbeats[t]*pointRatio), (curDownbeats[len(curDownbeats)-1]*pointRatio)
        curPhrase = curData[:, int(sFrame):int(eFrame)]
        #Perform local zero-mean unit-var
        curPhrase = (curPhrase - np.mean(curPhrase))/ np.max(curPhrase)
        # Resampling the phrases: here, each phrase will contain 4000 samples
        curPhrase = skt.resize(curPhrase, (nbBins, 4000), mode = 'reflect')
        measures_dict.append(curPhrase)#, curGenre, genreStr])
        meta_dict.append(curMeta)
    
    return measures_dict, meta_dict