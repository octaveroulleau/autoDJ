#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Downbeat detection

@author: pierre-amaury
"""

import madmom
import os

audio_file = "/home/pierre-amaury/Documents/PAM/projet/autodj_sets/datasets/gtzan/data/au/disco/disco.00000.au"

proc = madmom.features.RNNDownBeatProcessor()
probs = proc(audio_file)

#%%
beats = [[], []]
downbeats = [[], []]

thres = 0.4
Ts = 1.0/100

for i in range(1, probs.shape[0], 2):
    if probs[i, 0] > thres:
        beats[0].append(i*Ts)
        beats[1].append(probs[i, 0])
    if probs[i, 1] > thres:
        downbeats[0].append(i*Ts)
        downbeats[1].append(probs[i, 1])
        
print(len(beats[0]))
print(len(downbeats[0]))