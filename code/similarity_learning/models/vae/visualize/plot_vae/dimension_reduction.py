#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 19:19:14 2017

@author: chemla
"""
#import torch
#from torch.autograd import Variable

import torch
from torch.autograd import Variable
import numpy as np
import sklearn.manifold as manifold
import sklearn.decomposition as decomposition
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from numpy.random import permutation

def latent_dimselect(dims):
    def dimselect(z, *args, **kwargs):
        Z = np.zeros((z.shape[0], len(dims)))
        for i in range(len(dims)):
            Z[:,i] = z[:, dims[i]]
            Z[:,i] = z[:, dims[i]]
        return Z, dimselect
    return dimselect

def latent_isomap(z, *args, **kwargs):
    embedding = manifold.Isomap(*args, **kwargs)
    return embedding.fit_transform(z), embedding

def latent_lle(z, n_components=2, method='standard', *args, **kwargs):
    embedding = manifold.LocallyLinearEmbedding(n_components=n_components, method=method, *args, **kwargs)
    return embedding.fit_transform(z), embedding

def latent_mds(z, n_components=2, n_int=1, max_iter=100, *args, **kwargs):
    embedding = manifold.LocallyLinearEmbedding(n_components=n_components, n_int=n_int, max_iter=max_iter, *args, **kwargs)
    return embedding.fit_transform(z), embedding

def latent_tsne(z, n_components=2, init='pca', random_state=0, *args, **kwargs):
    embedding = manifold.TSNE(n_components=2, init='pca', random_state=0, *args, **kwargs)
    return embedding.fit_transform(z), embedding

def latent_pca(z, n_components=2, whiten=False, *args, **kwargs):
    embedding = decomposition.PCA(n_components=n_components, whiten=whiten, *args, **kwargs)
    return embedding.fit_transform(z), embedding

def plot_latent2(data, model, labels=None, n_points = None, method = latent_pca, write=None,
                 sampling=False, layer=0, color_map="plasma", zoom=10, *args, **kwargs):
    def get_cmap(n, color_map=color_map):
        return plt.cm.get_cmap(color_map, n)
    
    if n_points!=None:
        ids = permutation(data.shape[0])[0:n_points]
    else:
        ids = np.arange(data.shape[0])
    # print(data[ids,:])
    data = Variable(torch.from_numpy(np.abs(data[ids, :])), volatile=True)
    out = model.forward(data)
    if not sampling:
        if issubclass(type(out[1]), list):
            Z = out[1][layer][0].data.numpy()
            s = (torch.exp(torch.mean(out[1][layer][1], 1))*zoom).data.numpy()
        else:
            Z = out[1][0].data.numpy()
            s = (torch.exp(torch.mean(out[1][1], 1))*zoom).data.numpy()
    else:
        Z = out[2].data.numpy()
        s = np.ones(Z.shape[0])*zoom
        
    if Z.shape[1]>=2:
        print("[plot_laten2]-- dim(z) > 2. processing to dimension reduction...")
        Z, embedding = method(Z, *args, **kwargs)
        
    if not labels is None:
        meta = labels[ids, :]
        cmap = get_cmap(np.max(meta))
        c = []
        for i in meta:
            c.append(cmap(int(np.argwhere(i)[0])))
    else:
        c = 'b' 

    plt.scatter(Z[:, 0], Z[:,1], c=c, s=s)
    plt.show()



def plot_latent3(dataset, model, n_points = None, method = latent_pca, write=None,
                 task=None, sampling=False, layer=0, color_map="plasma", zoom=10, *args, **kwargs):
    def get_cmap(n, color_map=color_map):
        return plt.cm.get_cmap(color_map, n)
    
    if n_points!=None:
        ids = permutation(dataset.data.shape[0])[0:n_points]
    else:
        ids = np.arange(dataset.data.shape[0])
   
    data = Variable(torch.from_numpy(np.abs(dataset.data[ids, :])), volatile=True)
    out = model.forward(data)
    if not sampling:
        if issubclass(type(out[1]), list):
            Z = out[1][layer][0].data.numpy()
            s = (torch.exp(torch.mean(out[1][layer][1], 1))*zoom).data.numpy()
        else:
            Z = out[1][0].data.numpy()
            s = (torch.exp(torch.mean(out[1][1], 1))*zoom).data.numpy()
    else:
        Z = out[2].data.numpy()
        s = np.ones(Z.shape[0])*zoom
        
    if Z.shape[1]>=3:
        print("[plot_laten2]-- dim(z) > 2. processing to dimension reduction...")
        Z, _ = method(Z, n_components=3, *args, **kwargs)
        
    if task!=None:
        global caca
        meta = np.array(dataset.metadata[task])[ids]
        #print(meta, meta.shape, np.max(meta))
        cmap = get_cmap(np.max(meta))
        c = []
        for i in meta:
            c.append(cmap(int(i)))
    else:
        c = 'b'  
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(Z[:, 0], Z[:,1], Z[:, 2], c=c, s=s)



