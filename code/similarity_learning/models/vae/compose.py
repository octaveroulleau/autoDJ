""" ########### Running mode : composes a new track. ###############

Performs a "Random walk" in the VAE embedding space (with small cumulated distance)
Returns a list of mixing points : that can be used to re-synthetize a new track.
1. Define constraints for composition : number of chunks, probability of switching track...
2. Access the embedding space and perform a random walk with pre-defined constraints : track matching
3. From the chunk’s labels returned, create a list of mixing points

@author: laure

"""

import numpy as np
from numpy.random import permutation
from scipy.spatial.distance import cdist
import torch
from torch.autograd import Variable

import sys
sys.path.append('similarity_learning/models/vae/')

import visualize.plot_vae.dimension_reduction as dr
import VAE

import mixing_point as mp

import matplotlib.pyplot as plt

def compose_line(data):
	""" Here the composition is made by drawing a line in the VAE's latent space.
	Then the line is discretized and the nearest neighbor of each point is selected.

	Parameters
	----------
	data : numpy array
		A nb_chunks x dim_latent_space_cnn 2D array containing the chunks after a forward pass in the CNN.
		
	Returns
	-------
	idx_nearest_chunks : int list
		The indexes in "data" of the points that belong to the line (our composition path).

	Example
	-------

	import similarity_learning.models.vae.compose as vae_comp
	idx_nearest_chunks = vae_comp.compose_line(data)

	"""

	# Define constraints for composition
	nb_chunks_mix = 10

	# Load a pre-trained VAE
	filepath = 'similarity_learning/models/vae/saved_models/test_spec_softplus.t7'
	vae = VAE.load_vae(filepath)

	# Perform a forward pass to project data to the embedding space
	x = Variable(torch.from_numpy(data))
	x_params, z_params, z  = vae.forward(x)
	embedded_data = z[-1].data.numpy()
	dim_embedd_space = embedded_data.shape[1]
	nb_chunks_total = embedded_data.shape[0]

	# Random walk parameters : here, the parameters of a line.
	# First select two random datapoints as support for the trajectory
	idx_a = np.random.randint(nb_chunks_total)
	idx_b = np.random.randint(nb_chunks_total)
	if (idx_a == idx_b) :
		# Make sure idx_b != idx_a
		idx_b = (idx_a + 1) % nb_chunks_total
	a = embedded_data[idx_a,:]
	b = embedded_data[idx_b,:]
	discrete_line = create_discrete_line(dim_embedd_space, nb_chunks_mix,a,b)
	
	# For each point of the line, find its nearest neighbor in the embedded dataset
	idx_nearest_chunks = np.argmin(cdist(discrete_line,embedded_data),1) 
	# test_create_line(nb_chunks_mix,a[:3],b[:3])
	return idx_nearest_chunks

def create_discrete_line(dim_embedd_space, nb_chunks_mix,a,b):
	""" From two points, creates a segment in a high dimensional space.

	Parameters
	----------
	dim_embedd_space : int
		The dimension of the VAE's latent space
	nb_chunks_mix : int
		The total number of datapoints in the latent space
	a, b : np arrays of length dim_embedd_space
		The two datapoints that define the line

	Returns
	-------
	discrete_line : np array of dimension nb_chunks_mix x dim_embedd_space
		A set of nb_chunks_mix points in the latent space that belong to the segment [a,b].
		This defines a straight path between them in the latent space.

	Example
	-------

	a = embedded_data[idx_a,:]
	b = embedded_data[idx_b,:]
	discrete_line = create_discrete_line(dim_embedd_space, nb_chunks_mix,a,b)

	"""

	# Sample nb_chunks_mix points from the line
	t = np.sort(np.random.rand(nb_chunks_mix))
	t = np.tile(t,(dim_embedd_space,1))
	discrete_line = a[:,np.newaxis]*t + b[:,np.newaxis]*(1-t)
	discrete_line = np.transpose(discrete_line)
	return discrete_line

def test_create_line(nb_chunks_mix,a,b):
	""" From two points, visually check that the set generated is a line between them (in 3D only)

	Parameters
	----------
		nb_chunks_mix : int
			The total number of datapoints in the latent space
		a, b : np arrays of length dim_embedd_space
			The two datapoints that define the line

	Returns
	-------
	None. Plots a 3D representation of the line.

	Example
	-------

	a = embedded_data[idx_a,:][:3]
	b = embedded_data[idx_b,:][:3]
	discrete_line = create_discrete_line(dim_embedd_space, nb_chunks_mix,a,b)

	"""

	dim_embedd_space = 3
	discrete_line = create_discrete_line(dim_embedd_space, nb_chunks_mix,a,b)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	plt.scatter(discrete_line[0,:], discrete_line[1,:], discrete_line[2,:])
	plt.show()

def chunks_to_mp(idx_nearest_chunks, chunks_list, audioSet):
	"""
	Parameters
	----------
	idx_nearest_chunks : int np array
		The indexes of the chunks composing the mix, in the right temporal order.
	chunks_list : chunk_list object (see ...)
		The list of metadata related to all chunks
	audioSet : high-level object that manages the dataset.

	Returns
	-------
	mixing_points
		A list of mixing points (see mixing_point.py) to compose the new mix.

	Example
	-------

	idx_nearest_chunks = vae_comp.compose_line(data)
	mixing_points = chunks_to_mp(idx_nearest_chunks, chunks_list)
	finalset, sr = re.compose_track(mp_list)
	
	"""

	mixing_points = []
	previous_track_id = chunks_list.list_of_chunks[idx_nearest_chunks[0]].track_id # no mixing point at the beginning
	previous_ech_debut = chunks_list.list_of_chunks[idx_nearest_chunks[0]].ech_debut
	previous_ech_fin = -1
	previous_tempo = chunks_list.list_of_chunks[idx_nearest_chunks[0]].get_tempo(audioSet)

	for i in idx_nearest_chunks:
		track_id = chunks_list.list_of_chunks[i].track_id
		ech_debut = chunks_list.list_of_chunks[i].ech_debut
		ech_fin = chunks_list.list_of_chunks[i].ech_fin
		tempo = chunks_list.list_of_chunks[i].get_tempo(audioSet)

		if track_id != previous_track_id :
			previous_name = audioSet.files[previous_track_id]
			name = audioSet.files[track_id]
			mpi = mp.MixingPoint(previous_name, previous_ech_fin, previous_tempo, name, ech_debut, tempo)
			mixing_points.append(mpi)
			previous_track_id = track_id
			previous_ech_debut = ech_debut
			previous_ech_fin = ech_fin
			previous_tempo = tempo
		else :
			previous_ech_fin = ech_fin
		
	try:

		if (any(mp.tempo1 <= 0 for mp in mixing_points) or any(mp.tempo2 <= 0 for mp in mixing_points)):
			raise ValueError("A negative tempo was found when creating the mp list")

		if (any(mp.time1 < 0 for mp in mixing_points) or any(mp.time2 < 0 for mp in mixing_points)):
			raise ValueError("A negative time was found when creating the mp list")

	except ValueError as error:
		print(error)
	
	return mixing_points

