#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 21:16:42 2017

@author: chemla
"""

import torch
from torch.nn import functional as F
from torch.autograd import Variable

from numpy import pi, log
import pyro.distributions as dist

import sys
sys.path.append('../..')


# Probabilities
    
def LogBernoulli(x, x_params):
    return F.binary_cross_entropy(x_params[0], x, size_average = False)

def LogNormal(x, x_params, logvar=True, clamp=True):
    mean, std = x_params
    if not logvar:
        std = std.log()
    #std = torch.clamp(std, min=-30)
    result = torch.sum(0.5*(std + (x-mean).pow(2).div(std.exp())+log(2*pi)))
    return result

def get_crit(in_dist):
    if in_dist.dist_class==dist.bernoulli.dist_class:
        return LogBernoulli
    elif in_dist.dist_class==dist.normal.dist_class or in_dist.dist_class==cust.spectral.dist_class:
        return LogNormal
    else:
        raise Exception("Cannot find a criterion for distribution %s"%in_dist.dist_class)
    
    
# Kullback-Leibler Divergence

def GaussianKLDivergence(gauss1, prior=None, logvar=True):
    mean1, std1 = gauss1
    if not logvar:
        std1 = torch.log(std1)
    if prior==None:
        mean2, std2 = (Variable(gauss1[0].data.clone().zero_()), Variable(gauss1[1].data.clone().zero_()))
    else:
        mean2, std2 = prior['params']
        if not logvar:
            std2 = torch.log(std2)
    result = std2 - std1 + torch.exp(std1-std2) + torch.pow(mean1-mean2,2)/torch.exp(std2)
    result = torch.sum(result-1)*0.5
    return result

def get_kld(dist1, dist2):
    if dist1.dist_class==dist2.dist_class==dist.normal.dist_class:
        return GaussianKLDivergence
    else:
        raise Exception("Don't find KLD module for distributions %s and %s"%(dist1.dist_type, dist2.dist_type))


# Evidence Lower-Bound
    
class ELBO(object):
    def __init__(self, pinput, platent, plabel=None, sample=False, latent_params=None, *args, **kwargs):
        # Log-likelihood
        self.x_crit = []
        if not issubclass(type(pinput), list):
            pinput = [pinput]
        self.pinput = pinput
        for i in range(len(self.pinput)):
            self.x_crit.append(get_crit(self.pinput[i]['dist']))
        
        # Latent criterions
        self.z_crit = []
        if not issubclass(type(platent), list):
            platent =  [platent]
        self.platent = platent
        for i in range(len(self.platent)):
            if "prior" in self.platent[i]:
                prior = self.platent[i]["prior"]["dist"]
            else:
                prior = self.platent[i]["dist"]
            self.z_crit.append(get_kld(self.platent[i]["dist"], prior))
            
        
    def __call__(self, x, x_params, z_params, beta=1.0):
    # Log-likelihood
        rec_loss = 0.
        if not issubclass(type(x), list):
            x = [x]
        if not issubclass(type(x_params), list):
            x_params = [x_params]
        for i in range(len(x_params)):
            rec_loss += self.x_crit[i](x[i], x_params[i])
    # KLD error
        loss_kld = 0.
        if not issubclass(type(z_params), list):
            z_params = [z_params]
        for i in range(len(z_params)):
            if "prior" in self.platent[i]:
                prior = self.platent[i]["prior"]
            else:
                prior = None
            loss_kld += self.z_crit[i](z_params[i], prior=prior)
            
    # Global loss
        global_loss=rec_loss+beta*loss_kld
        return global_loss, rec_loss, loss_kld
 

