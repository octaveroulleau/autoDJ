import sys
sys.path.append("re_synthesis/")
sys.path.append("similarity_learning/models/vae/")
# sys.path.append("similarity_learning/")

import os
import librosa
import numpy as np
import mixing_point as mp    # from similarity_learning.models.vae 
import piece_of_track as pot # from similarity_learning.models.vae 
import assemble_mixing_points as re # from re_synthesis 

DATA_PATH = "../../../autodj_sets/datasets/gtzan/data/au/hiphop/"
DATA_NAMES = [DATA_PATH+'/'+f for f in os.listdir(DATA_PATH) if not f.startswith('.')]

#Toy MixingPoints definition
mp1 = mp.MixingPoint(DATA_NAMES[0],200000,120,DATA_NAMES[2],80000,90)
mp3 = mp.MixingPoint(DATA_NAMES[2],500000,120,DATA_NAMES[3],150000,120)
mp4 = mp.MixingPoint(DATA_NAMES[3],250000,120,DATA_NAMES[4],550000,120)
mp_list = [mp1,mp3,mp4]


finalset = re.compose_track(mp_list)

re.write_track(np.array(finalset))


