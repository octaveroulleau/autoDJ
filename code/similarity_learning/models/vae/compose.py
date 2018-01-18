""" Running mode : composes a new track.
Performs a "Random walk" in the VAE embedding space (with small cumulated distance)
Returns a list of mixing points : that can be used to re-synthetize a new track.
1. Define constraints for composition : number of chunks, probability of switching track...
2. Access the embedding space and perform a random walk with pre-defined constraints : track matching
3. From the chunk’s labels returned, create a list of mixing points
"""

# Random walk : draw a line in latent space, discretize, find nearest neighbors.

import numpy as np
from numpy.random import permutation
from scipy.spatial.distance import cdist
import torch
from torch.autograd import Variable

import sys
sys.path.append('similarity_learning/models/vae/')

import visualize.plot_vae.dimension_reduction as dr
import VAE

import matplotlib.pyplot as plt

def compose_line(data):
	""" Here the composition is made by drawing a line in the VAE's latent space.
	Then the line is discretized and the nearest neighbor of each point is selected.
	"""
	# Define constraints for composition
	nb_chunks_mix = 6

	# Load a pre-trained VAE
	filepath = 'similarity_learning/models/vae/saved_models/test_spec_softplus.t7'
	vae = VAE.load_vae(filepath)

	# Perform a forward pass to project data to the embedding space
	x = Variable(torch.from_numpy(data))
	x_params, z_params, z  = vae.forward(x)
	embedded_data = z[-1].data.numpy()

	# Random walk parameters : here, the parameters of a line.
	# First select two random datapoints as support for the trajectory
	dim_embedd_space = embedded_data.shape[1]
	nb_chunks_total = embedded_data.shape[0]
	# TODO : Check que les 2 sont differents !!!
	a = embedded_data[np.random.randint(nb_chunks_total),:]
	b = embedded_data[np.random.randint(nb_chunks_total),:]
	discrete_line = create_discrete_line(dim_embedd_space, nb_chunks_mix,a,b)
	
	# For each point of the line, find its nearest neighbor in the embedded dataset
	idx_nearest_chunks = np.argmin(cdist(discrete_line,embedded_data),1) 
	# test_create_line(nb_chunks_mix,a[:3],b[:3])
	return idx_nearest_chunks

def create_discrete_line(dim_embedd_space, nb_chunks_mix,a,b):
	# From two points a and b, returns nb_chunks_mix datapoints belonging to the segment [a,b].
	# Sample nb_chunks_mix points from the line
	t = np.sort(np.random.rand(nb_chunks_mix))
	t = np.tile(t,(dim_embedd_space,1))
	discrete_line = a[:,np.newaxis]*t + b[:,np.newaxis]*(1-t)
	discrete_line = np.transpose(discrete_line)
	return discrete_line

def test_create_line(nb_chunks_mix,a,b):
	dim_embedd_space = 3
	discrete_line = create_discrete_line(dim_embedd_space, nb_chunks_mix,a,b)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	plt.scatter(discrete_line[0,:], discrete_line[1,:], discrete_line[2,:])
	plt.show()

def chunks_to_mp(idx_nearest_chunks, chunks_list):
	mixing_points = []
	return mixing_points