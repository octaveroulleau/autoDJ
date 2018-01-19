#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:20:31 2017

@author: chemla
"""

from . import VanillaVAE as vae
from . import variational_modules as vmod
from functools import reduce




class AdversarialVAE(vae.VanillaVAE):
    def __init__(self, input_dim, latent_dim, input_dist, latent_dist, hidden_dim = 800, hidden_layers=2, *args, **kwargs):
        
        
    def init_modules(self, input_dim, hidden_dims, latent_dims, *args, **kwargs):
        super(AdversarialVAE, self).init_modules()
        self.discriminator = vmod.AdversarialLayer(latent_dims[-1], hidden_dims[-1])
        
    