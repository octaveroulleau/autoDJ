#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 18:38:11 2017

@author: chemla
"""

import torch.nn as nn
import torch.optim
from torch.autograd import Variable

import pyro
import pyro.optim
from pyro.infer import SVI
from pyro.util import ng_zeros, ng_ones
import pyro.distributions as dist

from criterions.ELBO import ELBO
from collections import OrderedDict
          
from .variational_modules import VariationalLayer

import pdb


class VanillaVAE(nn.Module):
    # initialisation of the VAE
    def __init__(self, input_params, latent_params, hidden_params = {"dim":800, "nlayers":2}, *args, **kwargs):
        if not hasattr(self, 'constructor'):
            self.constructor = {'input_params':input_params, 'latent_params':latent_params,
                                'hidden_params':hidden_params, 'args':args, 'kwargs':kwargs} # remind construction arguments for easy load/save
        super(VanillaVAE, self).__init__()
        self.pinput = input_params; self.phidden = hidden_params; self.platent = latent_params
        self.init_modules(self.pinput, self.phidden, self.platent, *args, **kwargs)
        self.manifolds={}
        
    def init_modules(self, input_params, hidden_params, latent_params, *args, **kwargs):
        self.encoders = type(self).make_encoder(input_params, hidden_params, latent_params, *args, **kwargs)
        self.decoders = type(self).make_decoder(input_params, hidden_params, latent_params, *args, **kwargs)
        
    @classmethod
    def make_encoder(cls, pinput, phidden, platent, *args, **kwargs):
        return nn.ModuleList([VariationalLayer(pinput, platent, phidden, nn_lin="ReLU", name="vae_encoder")])
                                 
    @classmethod
    def make_decoder(cls, pinput, phidden, platent, *args, **kwargs):
        phidden_dec = dict(phidden); phidden_dec['batch_norm'] = False
        return nn.ModuleList([VariationalLayer(platent, pinput, phidden, nn_lin="ReLU", name="vae_decoder")])
        
    # def guide(self, x):
    #     raise NotImplementedError("don't use it plzz")
    #     pyro.module("encoder", self.encoders)
    #     z_params = self.encoder.forward(x)
    #     if type(z_params)!=tuple:
    #         z_params = (z_params, )
    #     if self.distributions["latent"] == dist.normal:
    #         z_params = list(z_params)
    #         z_params[1] = torch.exp(z_params[1])
    #         z_params = tuple(z_params)
    #     pyro.sample("latent", self.lparams["dist"], *z_params)
        
    # processing methods
    def encode(self, x):
        z_params = self.encoders[0].forward(x)
        return z_params
    
    def decode(self, z):
        x_params = self.decoders[0].forward(z)
        return x_params
    
    def forward(self, x, pyro_model=False):
        z_params = self.encode(x)
        z = self.platent['dist'](*z_params)
        x_params = self.decode(z)
        return x_params, z_params, z
    
    # define optimizer (casual or pyro)
    def init_optimizer(self, usePyro=False, alg='Adam', optimArgs={"lr": 0.0001}, iw=1):
        self.usePyro = usePyro
        if usePyro:
            optimizer = getattr(pyro.optim, 'Adam')(optimArgs)
            self.optimizer = SVI(self.model, self.guide, optimizer, loss="ELBO")
        else:
            self.optimizer = getattr(torch.optim, 'Adam')(self.parameters(), **optimArgs)
            self.loss = ELBO(self.pinput, self.platent)
            
            
    def step(self, x, epoch, verbose=True, beta=1.0, warmup = 1., useCuda=False):
        if self.usePyro:
            lowerbound = self.optimizer.step(x)
            return lowerbound
        else:
            beta = beta*epoch/warmup
            self.optimizer.zero_grad()
            x_params, z_params, z = self.forward(x)
            # pdb.set_trace()
            global_loss, rec_loss, kld_loss = self.loss(x, x_params, z_params, beta=beta)
            if verbose:
                print("rec_loss : %f, kld_loss : %f"%(rec_loss.data.numpy(), kld_loss.data.numpy()))
            global_loss.backward()
            self.optimizer.step()
            if useCuda:
                global_loss = global_loss.cpu()
            return global_loss.data.numpy()

    def validate(self, x, epoch, verbose=True, beta=1.0, warmup = 1., useCuda=False):
            beta = beta*epoch/warmup
            x_params, z_params, z = self.forward(x)
            global_loss, rec_loss, kld_loss = self.loss(x, x_params, z_params, beta=beta)
            if verbose:
                print("val_loss : %f"%(global_loss.data.numpy()))
            if useCuda:
                global_loss = global_loss.cpu()
            return rec_loss.data.numpy()
        
    def save(self, filename, withDataset=False, cuda=False):
        if cuda:
            state_dict = OrderedDict(self.state_dict())
            for i, k in state_dict:
                k.cpu()
        else:
            state_dict = self.state_dict()
        constructor = dict(self.constructor)
        save = {'state_dict':state_dict, 'init_args':constructor, 'class':self.__class__, 'manifolds':self.manifolds}
        if withDataset!=False:
            save['dataset'] = withDataset
        torch.save({'state_dict':state_dict, 'init_args':constructor, 'class':self.__class__, 'manifolds':self.manifolds}, filename)
        
    @classmethod
    def load(cls, pickle):
        vae = cls(**pickle['init_args'])
        vae.load_state_dict(pickle['state_dict'])
        vae.manifolds = pickle['manifolds']
        return vae
        

            
