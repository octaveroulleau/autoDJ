# -*- coding: utf-8 -*-
"""

    The ``transformsSpectral`` module
    ========================
 
    This module builds various data augmentation transforms.
    The entire coding style approach is mimicked from Pytorch
    Here we define the simplest transforms and most importantly
    some meta-transforms used across different types

    Example
    -------
 
    Currently implemented
    ---------------------
    
    * WarpUniform               : Add cropped sub-sequences (uniform warping)
    * WarpNonLinear             : Add (non-linear) temporal warped series
    
    Comments and issues
    -------------------
    None for the moment

    Contributors
    ------------
    Philippe Esling (esling@ircam.fr)
    
"""
import numpy as np
import scipy.signal as sps

class WarpUniform(object):
    """
    Add cropped sub-sequences (uniform warping)
    
    Args:
        factor (int): Percentage to warp. default: .2
    """

    def __init__(self, factor=.2):
        self.factor = factor

    def __call__(self, data):
        """
        Args:
            data (Numpy array): Matrix to be masked
        Returns:
            Numpy array: Masked matrix.
        """
        maxCrop = data.shape[1] * self.factor
        cropLeft = np.ceil(np.random.rand() * maxCrop)
        cropRight = data.shape[1] - np.ceil(np.random.rand() * maxCrop)
        inputSeries = np.zeros((data.shape[0], int(cropRight - cropLeft + 1)))
        inputSeries[:, :] = data[:, int(cropLeft):int(cropRight)+1]
        data = sps.resample(inputSeries, data.shape[1]);
        return data;

class WarpNonLinear(object):
    """
    Add (non-linear) temporal warped series
    
    Args:
        factor (int): Deviation factor to the original size. default: .2
    """

    def __init__(self, factor=.2):
        self.factor = factor

    def __call__(self, data):
        """
        Args:
            data (Numpy array): Matrix to be masked
        Returns:
            Numpy array: Masked matrix.
        """
        switchPosition = np.floor(np.random.rand() * (data.shape[1] / 2) * self.factor) + (data.shape[0] / 2)
        leftSide = np.zeros((data.shape[0], int(switchPosition)))
        rightSide = np.zeros((data.shape[0], int(data.shape[1] - switchPosition)))
        leftSide[:, :] = data[:, :int(switchPosition)];
        rightSide[:, :] = data[:, int(switchPosition):]
        direction = np.random.rand() - 0.5 > 0 and 1 or -1
        leftSide = sps.resample(leftSide, int(switchPosition - (direction * 10)));
        rightSide = sps.resample(rightSide, int(data.shape[1] - switchPosition + (direction * 10)));
        print(leftSide.shape)
        print(rightSide.shape)
        tmpSeries = np.concatenate([leftSide, rightSide])
        return tmpSeries