#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Downbeat detection

@author: pierre-amaury
"""

import madmom

def downbeat_detection(track_path, beats_per_bar = 4):
    proc = madmom.features.DBNDownBeatTrackingProcessor(beats_per_bar, fps=100)
    activations = madmom.features.RNNDownBeatProcessor()(track_path)
    beats_position = proc(activations)
    
    downbeats = []
    for b in beats_position:
        if b[1] == 1.0:
            downbeats.append(b[0])
            
    return downbeats

def filter_multiple_occ(downbeats):
    filtered_downbeats = [downbeats[0]]
    for i in range(1, len(downbeats)):
        if not(abs(downbeats[i] - downbeats[i-1]) <= 1.0):
            filtered_downbeats.append(round(downbeats[i], 6))
    
    return filtered_downbeats