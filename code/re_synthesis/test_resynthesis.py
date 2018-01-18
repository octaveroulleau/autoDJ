import sys
sys.path.append("/Users/cyranaouameur/Desktop/ATIAM/PAM/Projet PAM/2017/autoDJ/code")

import os
import librosa
import numpy as np
from similarity_learning.models.vae import mixing_point as mp
from similarity_learning.models.vae import piece_of_track as pot
from re_synthesis import assemble_mixing_points as re

DATA_PATH = "/Users/cyranaouameur/Desktop/ATIAM/PAM/Projet PAM/2017/autodj_sets/datasets/mixtest"
DATA_NAMES = [DATA_PATH+'/'+f for f in os.listdir(DATA_PATH) if not f.startswith('.')]

#Toy MixingPoints definition
mp1 = mp.MixingPoint(DATA_NAMES[0],80000,120,DATA_NAMES[0],80000,160)
mp2 = mp.MixingPoint(DATA_NAMES[0],150000,160,DATA_NAMES[2],200000,90)
mp3 = mp.MixingPoint(DATA_NAMES[2],300000,90,DATA_NAMES[3],150000,150)
mp4 = mp.MixingPoint(DATA_NAMES[3],190000,150,DATA_NAMES[4],600000,110)
mp_list = [mp1,mp2,mp3,mp4]

#finalset, sr = re.compose_track(mp_list)
tracklist = re.fetch_audio(mp_list)
finalset, sr = re.mix_tracks(tracklist)
re.write_track(np.array(finalset),sr)


