#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Downbeat detection

@author: pierre-amaury
"""

import madmom
import mir_eval
import numpy as np

def downbeat_detection(track_path, beats_per_bar = 4):
    proc = madmom.features.DBNDownBeatTrackingProcessor(beats_per_bar, fps=100)
    activations = madmom.features.RNNDownBeatProcessor()(track_path)
    beats_position = proc(activations)
    
    downbeats = []
    for b in beats_position:
        if b[1] == 1.0:
            downbeats.append(b[0])
            
    return downbeats

def tempo_detection(track_path):
    proc = madmom.features.TempoEstimationProcessor(fps=100)
    activations = madmom.features.RNNBeatProcessor()(track_path)
    tempos = proc(activations) #list tempi with probability
    tempo = tempos[0][0]
    return tempo

def downbeat_evaluation(files_list, reference_downbeats, beats_per_bar = 4):
    f_measure_list = []
    for i in range(len(files_list)-1):
        print "====== File " + str(i) + " ======"
        estimated_downbeats = downbeat_detection(files_list[i], beats_per_bar)
#        print reference_downbeats[i]
#        print estimated_downbeats
        f_measure = mir_eval.beat.f_measure(reference_downbeats[i][0], np.array(estimated_downbeats))
#        print f_measure
        f_measure_list.append(f_measure)
    
    return f_measure_list

def filter_multiple_occ(downbeats):
    filtered_downbeats = [downbeats[0]]
    for i in range(1, len(downbeats)):
        if not(abs(downbeats[i] - downbeats[i-1]) <= 1.0):
            filtered_downbeats.append(round(downbeats[i], 6))
    
    return filtered_downbeats