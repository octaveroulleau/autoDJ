import sys
sys.path.append('pre_proccessing/')
sys.path.append('re_synthesis/')

import numpy as np
import skimage.transform as skt

from re_synthesis.const import SR
from pre_processing.chunkify import track_to_chunks

def preprocess_for_cnn(audioSet, audioOptions, nb_files):
	"""
	"""

	print("Loading audio data, please wait ...")

	for file_id in range(nb_files):

		downbeat = audioSet.metadata['downbeat'][file_id][0]
		chunks = track_to_chunks(file_id, SR, downbeat)
		data = []

		for i in range(len(chunks)):
			chunk = chunks[i].get_cqt(audioSet, audioOptions)
			nbBins = chunk.shape[0]
			chunk = skt.resize(chunk, (nbBins, 100), mode='reflect')
			data.append(chunk)

		if len(chunks) > 0 :

			x = np.swapaxes(np.array(data),1,2)

			if file_id == 0:
				X = x
			else:
				X = np.vstack((X,x))
		else :
			print("No chunks in file %d", file_id)

	return X