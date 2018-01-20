#####################################################################
# Here the file to run to execute the full process in evaluation mode
####################################################################

import numpy as np
import similarity_learning.models.vae.evaluate as vae_eval

""" 
#######################
# BEATS EVALUATION
#######################

beat_evaluation()

#######################
# DOWNBEATS EVALUATION
#######################

downbeat_evaluation()

# calculate score
downbeats_score_madmom = mir_eval_downbeats(downbeats_ref, downbeats_madmom)
"""

#######################
# CNN EVALUATION
#######################

# Feed the data forward in the CNN

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