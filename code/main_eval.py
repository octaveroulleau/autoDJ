###################
# Here the file to run to execute the full process in evaluation mode
###################

# Load the data and pre-process them

# Feed the data forward in the CNN

# Feed the data to the VAE

# Evaluate the model :
# T-SNE
# MIREVAL

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