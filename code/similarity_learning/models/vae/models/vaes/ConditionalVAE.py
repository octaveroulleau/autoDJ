#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 13:37:27 2017

@author: chemla
"""

from torch import cat
from . import VanillaDLGM 
from copy import deepcopy


class ConditionalVAE(VanillaDLGM):
    def __init__(self, input_params, latent_params, label_params, hidden_params = [{"dim":800, "layers":2}], *args, **kwargs):
        self.plabel = label_params
        super(ConditionalVAE, self).__init__(input_params, latent_params, hidden_params, plabel=label_params, *args, **kwargs)
        self.constructor['label_params'] = label_params
    
    @classmethod
    def make_encoder(cls, pinput, phidden, platent, plabel, *args, **kwargs):
        # add label to input of the encoder
        if not issubclass(type(pinput), list):
            enc_input = [pinput]
        else:
            enc_input = list(pinput)
        enc_input.append(plabel)
        return VanillaDLGM.make_encoder(enc_input, phidden, platent)
        
    @classmethod
    def make_decoder(cls, pinput, phidden, platent, plabel, *args, **kwargs):
        # add label to input of the decoder
        platent = deepcopy(platent)
        platent[-1]['dim'] += plabel['dim'] 
        return VanillaDLGM.make_decoder(pinput, phidden, platent)
    
    def forward(self, x, label_split=-1):
        ''''''
        z_params = self.encode(x)
        z = self.sample(z_params)
        z[-1] = cat((z[-1], x[label_split]), 1)
        x_params = self.decode(z)
        return x_params, z_params, z
        
    

        
        