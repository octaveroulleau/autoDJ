import sys
sys.path.append('pre_proccessing/')
sys.path.append('re_synthesis/')

import numpy as np
import skimage.transform as skt

from re_synthesis.const import SR
from pre_processing.chunkify import track_to_chunks
import chunkList as cl

def preprocess_for_cnn(audioSet, audioOptions, nb_files):
	""" Loads audio data to feed forward the neural networks.


	Parameters
    ----------
    audioSet : DatasetAudio object (see data/sets/audio)
    	Represents the dataset.
	audioOptions : dict
		The loading specifications for the dataset
	nb_files : int
		The desired number of audio files to load.

    Returns
    -------
	X : np array of size nb_chunks x [input_data_dims]

    Example
    -------

	audioSet, audioOptions = data.import_data.import_data()
	X = nns_data_load.preprocess_for_cnn(audioSet, audioOptions, len(audioSet.files))
	X_embed = np.asarray(model_cnn.predict(X, verbose = 1))

	TODO : it would actually be better not to do this operation each time. Instead,
	write the output to a pickle file and load directly the file.

	"""

	print("Loading audio data, please wait ...")
	custom_chunk_list = cl.ChunkList()

	for file_id in range(nb_files):

		downbeat = audioSet.metadata['downbeat'][file_id][0]
		chunks = track_to_chunks(file_id, SR, downbeat)
		data = []

		for i in range(len(chunks)):
			chunk = chunks[i].get_cqt(audioSet, audioOptions)
			nbBins = chunk.shape[0]
			chunk = skt.resize(chunk, (nbBins, 100), mode='reflect')
			data.append(chunk)
			custom_chunk_list.add_chunk(chunk)

		if len(chunks) > 0 :

			x = np.swapaxes(np.array(data),1,2)

			if file_id == 0:
				X = x
			else:
				X = np.vstack((X,x))
		else :
			print("No chunks in file %d", file_id)

	return X