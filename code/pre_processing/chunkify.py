#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Functions to create chunks from whole tracks

@author: pierre-amaury
"""

import chunkAudio as ca

def track_to_chunks(track_id, Fs, downbeat):
    """
    track_id : int of the track we want to chunk
    Fs : sampling frequency
    downbeat : array of int
    """
    chunks = [] # future list of chunks
    
    for i in range(downbeat.size-1):
        echantillon_debut = int(round(downbeat[i]*Fs))
        echantillon_fin = int(round(downbeat[i+1]*Fs))
        
        c = ca.ChunkAudio(Fs, track_id, echantillon_debut, echantillon_fin) #creation of the chunk
        chunks.append(c)
        
    return chunks