#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 18:21:29 2018

@author: pierre-amaury
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def compute_cqt(audio_file):
    y, sr = librosa.load(audio_file)
    C = np.abs(librosa.cqt(y, sr=sr, n_bins = 168, hop_length=1024, bins_per_octave=24, fmin = 64))
    return C