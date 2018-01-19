#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:42:51 2017

@author: chemla
"""

from . import VanillaVAE
from . import variational_modules as vmod
import pyro.distributions as dist
import torch.nn as nn

class SSVAE(VanillaVAE):
    def __init__(self, input_params, label_params, latent_params, hidden_params = {"dim":800, "layers":2}, *args, **kwargs):
        super(SSVAE, self).__init__(input_params, label_params, latent_params, hidden_params, *args, **kwargs)    
        if not hasattr(self, 'constructor'):
            self.constructor = {'input_params':input_params, 'label_params':label_params, 'latent_params':latent_params, 
                                'hidden_params':hidden_params, 'args':args, 'kwargs':kwargs} # remind construction arguments for easy load/save
        self.pinput = input_params; self.phidden = hidden_params; self.platent = latent_params; self.plabel = label_params
        self.init_modules(input_params, label_params, hidden_params, latent_params)
        
    def init_modules(self, pinput, phidden, platent, plabel, *args, **kwargs):
        self.encoder = nn.ModuleList([vmod.DeterministicLayer(pinput['dim'], phidden['dim'], n_layers=phidden["layers"]),
                                      vmod.ClassifierLayer(pinput['dim'], phidden['dim'], plabel['dim'], hidden_layers=1),
                                      vmod.get_module(platent["dist"])(phidden['dim'], phidden['dim'], platent['dim'], hidden_layers=0)])
        # TODO : merge layers of encoder and classifiers
        input_decoder = {'dim':pinput['dim'] + plabel['dim']}
        self.decoder = super(SSVAE, self).make_decoder(input_decoder, phidden, pinput, *args, **kwargs)
                                     
    def encode(self, x):
        h = self.encoder[0](x); y = self.encoder[1](h); z = self.encoder[2](h);
        z, y = self.classifier(x)
        return z,y
        
    def decode(self, z, y):
        x = self.decoder(z,y)
        return x
        
    def forward(self, x, y=None):
        z_params, y_params = self.encode(x)
        z = self.platent['dist'](*z_params)
        if y==None:
            y = self.plabel['dist'](*y_params)
        x_params = self.decode(z,y)
        if type(x_params)!=tuple:
            x_params = (x_params,)
        if type(z_params)!=tuple:
            z_params=(z_params, )
        if type(y_params)!=tuple:
            y_params=(y_params, ) 
        return x_params, (z_params,), y_params
        

            
        
        
        