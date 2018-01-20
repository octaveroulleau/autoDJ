#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:28:24 2017

@author: chemla


"""

#%%
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

from pyro import distributions as dist
from numpy.random import permutation
from models.vaes import VanillaVAE, VanillaDLGM, ConditionalVAE
		
def build_model(model_type, input_dim, label_dim=[0]):
	""" Allows one to easily build a vae model by selecting one of preset models.

	Parameters
    ----------
    model_type : int np array
        The indexes of the chunks composing the mix, in the right temporal order.
    input_dim : int
    	The dimension of the input data vectors
    label_dim : (optional) int
    	For VAEs that require labels, the dimension of the labels

    Returns
    -------
	use_label : bool
		A parameter for the training supervised mode).
	vae : 
		The VAE model to be trained.

    Example
    -------

	use_label, vae = VAE.build_model("dlgm", input_dim)
    
    """

	"""Salut l’ami,

Alors je te conseille effectivement de commencer par la et de toujours conserver le couple ReLU et warm-up, après pour les tests judicieux je dirais:
Nombre de couches = [1, 2]
Nombre de neurones = [800, 2000]

	"""

	if model_type == "vae": # for Vanilla VAE
		input_params = [{"dim":input_dim[0], "dist":dist.normal},{"dim":input_dim[1], "dist":dist.normal}]
		latent_params= {"dim":20, "dist":dist.normal}
		hidden_params= {"dim":1500, "nlayers":2, "batch_norm":False}
		vae = VanillaVAE(input_params, latent_params, hidden_params)
		use_label = False

	if model_type == "dlgm": # for Deep Latent Gaussian Models
		prior = {"dist":dist.normal, "params":(Variable(torch.Tensor(16).fill_(1), requires_grad=False),
											   Variable(torch.Tensor(16).zero_(), requires_grad=False))}
		input_params = {"dim":input_dim, "dist":dist.bernoulli}
		latent_params = [{"dim":64, "dist":dist.normal}, {"dim":32, "dist":dist.normal}, {"dim":16, "dist":dist.normal, "prior":prior}]
		hidden_params= [{"dim":800, "nlayers":2, "batch_norm":False},
						{"dim":400, "nlayers":2, "batch_norm":False},
						{"dim":200, "nlayers":2, "batch_norm":False}]
		vae = VanillaDLGM(input_params, latent_params, hidden_params)
		use_label = False

	if model_type == "cvae": # for Conditional Auto-encoders
		input_params = [{"dim":input_dim, "dist":dist.bernoulli},{"dim":input_dim, "dist":dist.bernoulli}]
		latent_params = [{"dim":64, "dist":dist.normal}, {"dim":32, "dist":dist.normal}, {"dim":16, "dist":dist.normal}]
		hidden_params= [{"dim":800, "nlayers":2, "batch_norm":True},
					{"dim":400, "nlayers":2, "batch_norm":True},
					{"dim":200, "nlayers":2, "batch_norm":True}]
		label_params = {"dim":label_dim, "dist":dist.categorical}
		vae = ConditionalVAE(input_params, latent_params, label_params, hidden_params)
		use_label = True

	if model_type == "autodj": # Custom architecture for PAM auoDJ
		prior = {"dist":dist.normal, "params":(Variable(torch.Tensor(16).fill_(1), requires_grad=False),
											   Variable(torch.Tensor(16).zero_(), requires_grad=False))}
		input_params = {"dim":input_dim, "dist":dist.bernoulli}
		latent_params = [{"dim":16, "dist":dist.normal, "prior":prior}]
		hidden_params= [{"dim":800, "nlayers":1, "batch_norm":False}]
		vae = VanillaDLGM(input_params, latent_params, hidden_params)
		use_label = False

	return use_label, vae

def load_vae(filepath):
	# to load an external model from 'filepath'
	loaded = torch.load(filepath)
	vae = loaded['class'].load(loaded)
	return vae

def train_vae(vae, data, max_epochs=100, batch_size=100, model_type="dlgm", labels=[], use_cuda=False):
	""" Trains the VAE model on the given dataset.

	Parameters
    ----------
    vae : vae model
        The indexes of the chunks composing the mix, in the right temporal order.
    data : 2D numpy array of size nb_chunks x dim_data
    	The matrix of the training data vectors

    Returns
    -------
	vae : vae model
		The model after training.

    Example
    -------

	vae = VAE.train_vae(vae, data, max_epochs, batch_size, model_type, use_cuda)

	"""
    
	epoch = -1
	# Beta = [0.5, 1, 4]
	beta = 4
	logs = [[],[]]

	while epoch < max_epochs:
		epoch += 1
		epoch_loss = 0.
		batch_ids = permutation(len(data)) 
		for i in range(len(batch_ids)//batch_size - 1):
			# load data
			x = Variable(torch.from_numpy(data[batch_ids[i*batch_size:(i+1)*batch_size]]).float())

			# not important stuff
			if model_type == "cvae":
				labels = Variable(torch.from_numpy(labels[batch_ids[i*batch_size:(i+1)*batch_size]]).float())
				x = [x, labels]
			# use cuda
			if use_cuda:
				x = x.cuda()
				
			# step
			batch_loss = vae.step(x, epoch, verbose=False, warmup=1, beta=beta)
			epoch_loss += batch_loss
			# print("epoch %d / batch %d / lowerbound : %f "%(epoch, i, batch_loss))

		# validate
		i = len(batch_ids)//batch_size - 1
		x_val = Variable(torch.from_numpy(data[batch_ids[i*batch_size:(i+1)*batch_size]]).float())
		val_loss = vae.validate(x_val, epoch, verbose=False, warmup=1, beta=beta)
		# print("epoch %d / val_loss : %f "%(epoch, val_loss))
			
		print("---- FINAL EPOCH %d LOSS : %f"%(epoch, epoch_loss))
		logs[0].append(epoch_loss[0])
		logs[1].append(val_loss[0])
		
	return vae, logs
		


