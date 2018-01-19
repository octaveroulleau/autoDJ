#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 21:55:58 2017

@author: chemla
"""

import numpy as np


def save_latent_grid(path, model, dataset, n, scale=1.0 ,reduction="pca"):
    frame=[[-scale,scale], [-scale, scale]]
    i_range = np.linspace(frame[0][0], frame[0][1], n)
    j_range = np.linspace(frame[1][0], frame[1][1], n)
    grid = torch.Tensor(n,n,dataset[0][0].size(1),dataset[0][0].size(2))
    
    for i in range(len(i_range)):
        for j in range(len(j_range)):
            if model.latent_dim==2:
                model_in = Variable(torch.from_numpy(np.array([i_range[i],j_range[j]])).float(), volatile=True)
            else:
                if reduction=="pca":
                    if model.pca ==[]:
                        raise Exception("for model with latent_dim>2, please provide a pca to plot a latent grid")
                    latent_coords = numpy.dot(np.array([i_range[i],j_range[j]]) ,model.pca.Wt[0:2])
                    latent_coords /= model.pca.fracs
                    latent_coords += model.pca.mu
                    model_in = Variable(torch.from_numpy(latent_coords).float(), volatile=True)
                elif reduction=="ISOMAP":
                    embedding = manifold.Isomap(*args)
                    zs = embedding.fit_transform(z)
                elif type(reduction)==tuple:
                    if len(reduction)==2:
                        latent_coords = torch.FloatTensor(model.latent_dim).fill_(0)
                        latent_coords[reduction[0]] = i_range[i]
                        latent_coords[reduction[1]] = j_range[j]
                        #latent_coords[1] = -1
                        model_in = Variable(latent_coords, volatile=True)
                    else:
                        raise ValueError("please give a 2d tuple for reduction")
            grid[i,j,:,:]=model.decode(model_in)[0].data
            
    img = make_grid(grid.view(n*n, dataset[0][0].size(1),dataset[0][0].size(2)).unsqueeze(1), nrow=n)
    save_image(img, path)