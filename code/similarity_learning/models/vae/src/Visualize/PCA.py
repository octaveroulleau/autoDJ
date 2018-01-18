"""PCA.py

This module gives functions for PCA analysis.
Unit testing: see ../unitTest/VAETest.py
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

#test = np.array([[1.1,2.3,3.7,4.3,5.8],[1.3,2.2,3.8,4.3,5.1],[1.2,2.9,3.6,4.5,5.4],[1.7,2.7,3.5,4.9,5.4],[1.2,2.4,3.1,4.8,5.2]])
#testlabels = np.array(['1','1','1','2','2']).transpose()
#
#X = np.loadtxt("mnist2500_X.txt");
#labels = np.loadtxt("mnist2500_labels.txt");


def PCA_reduction(dataX, dim=2):
    """PCA_reduction(data, dim=2) : This function's input is a Torch Tensor or a np.array containing high-dimensionality data
    and it returns the low dimensionality data (dim=2 by default) found with PCA algorithm"""
    # preprocess the data
    if isinstance(dataX, np.ndarray):
        dataX = torch.from_numpy(dataX)

    X = dataX
    X_mean = torch.mean(X, 0)
    X_centered = X - X_mean.expand_as(X)

    # compute covariance matrix
    cov = np.cov(X_centered.t().numpy())
    t_cov = torch.from_numpy(cov)

    # compute eigenvalues/ eigenvectors
    e, v = torch.symeig(t_cov, eigenvectors=True)

    # restrict the number of dimensions
    e_r = e[-dim:]
    v_r = v[:, -dim:]
    # transform into the new base
    out = torch.mm(X, v_r)

    return out, v_r


def PCA_vision(dataX, labels, lbldim, outDim=2):
    """PCA_vision(data, dim=2) : This function's input is a Torch Tensor or a np.array containing high-dimensionality data
    and it plots the low dimensionality data (dim=2 by default) found with PCA algorithm"""
    tensor = PCA_reduction(dataX, outDim)
    new = tensor[0].numpy()
    plt.scatter(new[:, 0], new[:, 1], 20, labels[
                lbldim, :])  # wouldn't work with dim!=2
    plt.legend()
    plt.show()
    return True
