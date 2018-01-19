#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:49:58 2017

@author: chemla
"""

from torch import cat
import torch.nn as nn
import pyro.distributions as dist
from collections import OrderedDict

import sys
sys.path.append('../..')

#%% Distribution layers



class GaussianLayer(nn.Module):
    '''Module that outputs parameters of a Gaussian distribution.'''
    def __init__(self, input_dim, output_dim):
        '''Args:
            input_dim (int): dimension of input
            output_dim (int): dimension of output
        '''
        super(GaussianLayer, self).__init__()
        self.input_dim = input_dim; self.output_dim = output_dim
        self.modules_list = nn.ModuleList()
        self.modules_list.append(nn.Linear(input_dim, output_dim))
        self.modules_list.append(nn.Linear(input_dim, output_dim))
        
    def forward(self, ins):
        '''Outputs parameters of a diabgonal Gaussian distribution.
        :param ins : input vector.
        :returns: (torch.Tensor, torch.Tensor)'''
        mu = self.modules_list[0](ins)
        logvar = self.modules_list[1](ins)
        return mu, logvar
    
class SpectralLayer(GaussianLayer):
    def __init__(self, input_dim, output_dim):
        super(SpectralLayer, self).__init__(input_dim, output_dim)
        self.modules_list[0] = nn.Sequential(self.modules_list[0], nn.Softplus())
        
        
class BernoulliLayer(nn.Module):
    '''Module that outputs parameters of a Bernoulli distribution.'''
    def __init__(self, input_dim, output_dim, label_dim=1):
        super(BernoulliLayer, self).__init__()
        self.input_dim = input_dim; self.output_dim = output_dim
        self.modules_list = nn.Sequential(nn.Linear(input_dim, output_dim), nn.Sigmoid())
        
    def forward(self, ins):
        mu = self.modules_list(ins)
        return (mu,) 

class CategoricalLayer(nn.Module):
    '''Module that outputs parameters of a categorical distribution.'''
    def __init__(self, input_dim, label_dim):
        super(CategoricalLayer, self).__init__()
        self.input_dim = input_dim; self.label_dim = label_dim
        self.modules_list = nn.Sequential(nn.Linear(input_dim, label_dim), nn.Softmax())
        
    def forward(self, ins):
        probs = self.modules_list(ins)
        return (probs,)
    
def get_module(distrib):
    if distrib.dist_class == dist.normal.dist_class:
        return GaussianLayer
    elif distrib.dist_class == dist.bernoulli.dist_class:
        return BernoulliLayer
    elif distrib.dist_class == dist.categorical.dist_class:
        return CategoricalLayer
    elif distrib.dist_class == cust.spectral.dist_class:
        return SpectralLayer
    else:
        raise TypeError('Unknown distribution type : %s'%distrib)
        
        
#%% Generic Layers

class VariationalLayer(nn.Module):
    ''' Generic layer that is used by generative variational models as encoders, decoders or only hidden layers.'''
    def __init__(self, pins, pouts=None, phidden={"dim":800, "nlayers":2}, nn_lin="ReLU", name=""):
        ''':param pins: Input properties.
        :type pins: dict or [dict]
        :param pouts: Out propoerties. Leave to None if you only want hidden modules.
        :type pouts: [dict] or None
        :param phidden: properties of hidden layers.
        :type phidden: dict
        :param nn_lin: name of non-linear layer 
        :type nn_lin: string
        :param name: name of module
        :type name: string'''
        # Configurations
        super(VariationalLayer, self).__init__()
        self.input_dim = 0; self.phidden = phidden;
        if not issubclass(type(pins), list):
            pins = [pins]
            
        for p in pins:
            self.input_dim += p['dim']
        
        # get hidden layers
        self.hidden_module = self.get_hidden_layers(self.input_dim, phidden, nn_lin, name)
        
        # get output layers
        self.latent_params = pouts
        if pouts!=None:
            self.out_modules = self.get_output_layers(phidden['dim'], pouts)
        else:
            self.out_modules=None
    
    def get_output_layers(self, in_dim, pouts):
        '''returns output layers with resepct to the output distribution
        :param in_dim: dimension of input
        :type in_dim: int
        :param pouts: properties of outputs
        :type pouts: dict or [dict]
        :returns: ModuleList'''
        out_modules=[]
        if not issubclass(type(pouts), list):
            pouts = [pouts]
        for p in pouts:
            out_modules.append(get_module(p["dist"])(in_dim, p["dim"]))
        return nn.ModuleList(out_modules)
    
    
    def forward(self, x, outputHidden=False):
        '''outputs parameters of corresponding output distributions
        :param x: input or vector of inputs.
        :type x: torch.Tensor or [torch.Tensor ... torch.Tensor]
        :param outputHidden: also outputs hidden vector
        :type outputHidden: True
        :returns: (torch.Tensor..torch.Tensor)[, torch.Tensor]'''
        if type(x)==list:
            ins = cat(x, 1)
        else:
            ins = x
        h = self.hidden_module(ins)
        z = []
        if self.out_modules!=None:
            for i in self.out_modules:
                z.append(i(h)) 
            if not issubclass(type(self.latent_params), list):
                z = z[0]
        if outputHidden:
            if self.out_modules!=None:
                return z,h
            else:
                return h
        else:
            if self.out_modules!=None:
                return z
            else:
                return h
    
    @classmethod
    def get_hidden_layers(cls, input_dim, phidden={"dim":800, "nlayers":2, "batch_norm":False}, nn_lin="ReLU", name=""):
        '''outputs the hidden module of the layer.
        :param input_dim: dimension of the input
        :type input_dim: int
        :param phidden: parameters of hidden layers
        :type phidden: dict
        :param nn_lin: non-linearity name
        :type nn_lin: str
        :param name: name of module
        :type name: str
        :returns: nn.Sequential'''
        # Hidden layers
        modules = OrderedDict()
        for i in range(phidden['nlayers']):
            if i==0:
                modules["hidden_%d"%i] =  nn.Linear(int(input_dim), int(phidden['dim']))
            else:
                modules["hidden_%d"%i] = nn.Linear(int(phidden['dim']), int(phidden['dim']))
            if phidden['batch_norm']:
                modules["batch_norm_%d"%i]= nn.BatchNorm1d(phidden['dim'])
            modules["nnlin_%d"%i] = getattr(nn, nn_lin)()
        return nn.Sequential(modules)
    
        
class DLGMLayer(nn.Module):
    ''' Specific decoding module for Deep Latent Gaussian Models'''
    def __init__(self, pins, pouts, phidden={"dim":800, "nlayers":2}, nn_lin="ReLU", name=""):
        '''
        :param pins: parameters of the above layer
        :type pins: dict
        :param pouts: parameters of the ouput distribution
        :type pouts: dict
        :param phidden: parameters of the hidden layer(s)
        :type phidden: dict
        :param nn_lin: non-linearity name
        :type nn_lin: str
        :param name: name of module
        :type name: str'''
        super(DLGMLayer, self).__init__()
        self.hidden_module = VariationalLayer.get_hidden_layers(pins["dim"], phidden=phidden, nn_lin=nn_lin, name=name)
        self.out_module = nn.Linear(phidden['dim'], pouts['dim'])
        self.cov_module = nn.Linear(pouts['dim'], pouts['dim'])
        
    def forward(self, z, eps):
        '''outputs the latent vector of the corresponding layer
        :param z: latent vector of the above layer
        :type z: torch.Tensor
        :param eps: latent stochastic variables
        :type z: torch.Tensor
        :returns:torch.Tensor'''
        if issubclass(type(z), list):
            z = torch.cat(tuple(z), 1)
        return self.cov_module(eps) + self.out_module(self.hidden_module(z))
        
    

        