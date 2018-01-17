#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 13:36:44 2017

@author: chemla
"""

from torch import Tensor
import torch.nn as nn
from torch.autograd import Variable
import pyro.distributions as dist
import pyro
from . import VanillaVAE 
from .variational_modules import VariationalLayer, DLGMLayer

class VanillaDLGM(VanillaVAE):
    def __init__(self, input_params, latent_params, hidden_params = {"dim":800, "layers":2}, *args, **kwargs):
        super(VanillaDLGM, self).__init__(input_params, latent_params, hidden_params, *args, **kwargs)
        
    # design routines
    @classmethod
    def make_encoder(cls, pinput, phidden, platent, *args, **kwargs):
        encoders = nn.ModuleList()
        for i in range(len(platent)):
            if i == 0:
                encoders.append(VariationalLayer(pinput, platent[i], phidden[i], name="dlgm_encoder_%d"%i))
            else:
                encoders.append(VariationalLayer(phidden[i-1], platent[i], phidden[i]))
        return encoders
            
    @classmethod    
    def make_decoder(cls, pinput, phidden, platent, *args, **kwargs):
        decoders = nn.ModuleList()
        #decoders.append(nn.Linear(platent[-1]['dim'], platent[-1]["dim"]))
        decoders.append(VariationalLayer(platent[-1], phidden={'dim':platent[-1]['dim'], "nlayers":1, "batch_norm":False}))
        for i in reversed(range(0, len(platent))):
            if i == 0:
                phidden_dec = dict(phidden[0]); phidden_dec['batch_norm']=False
                decoders.append(VariationalLayer(platent[0], pinput, phidden_dec, name="dlgm_decoder_%d"%i))
            else:
                decoders.append(DLGMLayer(platent[i], platent[i-1], phidden[i], name="dlgm_decoder_%d"%i))
        return decoders

         
    # Pyro routines
    def model(self, x):
        raise NotImplementedError("not working yet")
        n_layers = len(self.dims["latent"])
        zs = []
        for l in range(n_layers):
            z_mu = ng_zeros([x.size(0), self.lparams["dim"][l]], type_as=x.data)
            z_sigma = ng_ones([x.size(0), self.lparams["dim"][l]], type_as=x.data)
            z = pyro.sample("latent_%d"%l, self.lparams["dist"][l], z_mu, z_sigma)
            zs.append(z)
        decoder_out = self.decode(zs)
        if self.pinput["dist"] == dist.normal:
            decoder_out = list(decoder_out)
            decoder_out[1] = torch.exp(decoder_out[1])
            decoder_out = tuple(decoder_out)
        pyro.sample("obs", self.pinpit["dist"], obs=x.view(-1, self.pinput["dim"]), *decoder_out)
        
    def guide(self, x):
        raise NotImplementedError("not working yet")
        z_params = self.encode(x)
        for l in range(len(latent_params)):
            if self.platent["dist"][l] == dist.normal:
                params = list(z_params[l])
                params[1] = torch.exp(params[1])
                z_params = tuple(params)
                
            pyro.sample("latent_%d"%l, self.distributions["latent"][l], *z_params)

    # Process routines
    def encode(self, x):
        previous_output = x
        latent_params = []
        for i in range(0, len(self.platent)):
            params, h = self.encoders[i](previous_output, outputHidden=True)
            latent_params.append(params)
            previous_output = h
        return latent_params
    
    def decode(self, z, t=0):
        if not issubclass(type(z), list):
            z = [z]
        z = z[::-1]
        if t==0:
            previous_output = self.decoders[0](z[0])
            t+=1
        else:
            previous_output = z[0]
        for i in range(t, len(self.platent)):
            if (i-t+1)<len(z):
                eps = z[i-t+1]
            else:
                eps = Variable(Tensor(z[0].size(0), self.platent[-(i+1)]['dim']).zero_())
            previous_output = self.decoders[i](previous_output, eps)        
        x_params = self.decoders[-1](previous_output)
        return x_params
        
            
    def sample(self, z_params):
        z = [] 
        for i in range(len(self.platent)):
            z.append(self.platent[i]["dist"](*z_params[i]))
        return z
    
    def forward(self, x):
        z_params = self.encode(x)
        z = self.sample(z_params)
        x_params = self.decode(z)
        return x_params, z_params, z
    
    