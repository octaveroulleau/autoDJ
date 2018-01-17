""" Evaluation mode.
1. Defnition of a custom score : distance between two consecutive chunks of a track should be minimal
2. Evaluate score on the training dataset
3. Use T-SNE to visually check coherence of the embedding space
"""

# Use toymix !!! (chunk by chunk)

def forward(cnn_data, model):
	""" Input training set in CNN's feature space
	Output the same set in the VAE's latent space
	"""

def score_vae(model):
	"""
	For each track, find the path representing the chunks it is composed of in the latent space
	Cumulate length of all paths
	Print score of the model
	"""

def t_sne_train(model):
	"""
	Plots a representation of the training data in the latent space using t-sne dimensionality reduction.
	Plots each chunk along with its label.
	Plots tracks as paths.
	"""
