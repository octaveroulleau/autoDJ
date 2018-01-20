#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Functions to create chunks from whole tracks

@author: pierre-amaury
"""

import sys
import numpy as np
sys.path.append("pre_processing/")
sys.path.append("data/")
import pdb
import os
import chunkAudio as ca
import chunkList as cl
import skimage.transform as skt
from data.sets.audio import DatasetAudio


def load_dataset():
    """ Function that load the whole dataset
    
    Parameters
    ----------
    
    Returns
    -------
    DatasetAudio instance
        Access to metadata with : audioSet.metadata["task"] where taskcan be "artist", "beat", "downbeat", "genre", "key", "tempo".
    
    """
    baseFolder = "../../../autodj_sets/datasets"
    taskList = ["artist", "beat", "downbeat", "genre", "key", "tempo"]
    
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
            
    audioSet = DatasetAudio(audioOptions);
    audioSet.importMetadataTasks();
    
    return audioSet

def track_to_chunks(track_id, Fs, downbeat):
    """ Function that divides a track into multiple chunks according to its downbeats
    
    Parameters
    ----------
    track_id : int
        The id of the track to divide into chunks
    Fs : int
        Sampling frequency of the track n° track_id
    downbeat : float list
        Instants in seconds where downbeats appear in the track n° track_id
        
    Returns
    -------
    ChunkAudio list
        A list of ChunkAudio instances
    """
    chunks = [] # future list of chunks
    
    for i in range(len(downbeat)-1):
        echantillon_debut = int(round(downbeat[i]*Fs))
        echantillon_fin = int(round(downbeat[i+1]*Fs))
        
        c = ca.ChunkAudio(Fs, track_id, echantillon_debut, echantillon_fin) #creation of the chunk
        chunks.append(c)
        
    return chunks

def dataset_to_chunkList(audioSet, Fs):
    """ Function that transforms a DatasetAudio instance into a list of chunks (of all tracks of the dataset)
    
    Parameters
    ----------
    audioSet : DatasetAudio instance
        The DatasetAudio instance that can be load with load_dataset()
    Fs : int
        Sampling frequency of all tracks of the dataset
    
    """
    chunk_list = cl.ChunkList()
    downbeats = audioSet.metadata["downbeat"]
    L = len(downbeats)-1
    Fs = 22050
    for i in range(L):
        chunks = track_to_chunks(i, Fs, downbeats[i][0])
        for c in range(len(chunks)-1):
            chunk_list.add_chunk(chunks[c])
    
    return chunk_list
    