#####################################################################
# Here the file to run to execute the full process in evaluation mode
####################################################################

import numpy as np
import similarity_learning.models.vae.evaluate as vae_eval

""" 
#######################
# BEATS EVALUATION
#######################

# load reference data
beats_ref = load_beats_reference()

# compute beats with several algorithms
beats_madmom = compute_beats_madmom()
beats_librosa = compute_beats_librosa()
beats_ellis = compute_beats_ellis()

# calculate score
beats_score_madmom = mir_eval_beats(beats_ref, beats_madmom)
beats_score_librosa = mir_eval_beats(beats_ref, beats_librosa)
beats_score_ellis = mir_eval_beats(beats_ref, beats_ellis)

#########################
# DOWNBEATS EVALUATION
#########################

# load reference data
downbeats_ref = load_downbeats_reference()

# compute downbeats with several algorithms
downbeats_madmom = compute_downbeats_madmom()

# calculate score
downbeats_score_madmom = mir_eval_downbeats(downbeats_ref, downbeats_madmom)
"""

#######################
# CNN EVALUATION
#######################

# Feed the data forward in the CNN

# import matplotlib.pyplot as plt
# import pickle
# tsne = pickle.load(open('./tsne', 'rb'))
# #%%

# tsne = tsne[1:]

# plt.figure()
# for i in range(len(tsne)):
#     for j in range(len(tsne[i])):
#         plt.scatter(tsne[i][j][0][0], tsne[i][j][0][1] )

# plt.show()

# Evaluate the model :
# T-SNE
# MIREVAL

#######################
# VAE EVALUATION
#######################

# Fake dataset
input_dim = 1000
nb_chunks = 123
data = np.random.rand(nb_chunks,input_dim).astype('float32')
# Feed to the VAE and return indexes of nearest chunks
idx_nearest_chunks = vae_eval.evaluate(data)
# Evaluate the model :
# T-SNE
# MIREVAL