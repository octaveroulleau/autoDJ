# -*- coding: utf-8 -*-
"""

    The ``transforms`` module
    ========================
 
    This module builds various data augmentation transforms.
    The entire coding style approach is mimicked from Pytorch
    Here we define the simplest transforms and most importantly
    some meta-transforms used across different types

    Example
    -------
 
    Currently implemented
    ---------------------
    
    * Compose       : Composes several transforms together.
    * ComposeRandom : Composes several transforms in a random order.
    * Lambda        : Apply a user-defined lambda as a transform.
    * Scale         : Scale a tensor to a floating point between -1.0 and 1.0.
    * ToPytorch     : Convert ndarrays to Pytorch Tensors.
    * ToTensorflow  : Convert ndarrays to Tensorflow Tensors.
    
    Comments and issues
    -------------------
    None for the moment

    Contributors
    ------------
    Philippe Esling (esling@ircam.fr)
    
"""
#import torch
import numpy as np
#import tensorflow as tf
import types

class Compose(object):
    """
    Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.Scale(),
        >>>     transforms.PadTrim(max_len=16000),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

class ComposeRandom(object):
    """
    Composes several transforms together in a random order.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.Scale(),
        >>>     transforms.PadTrim(max_len=16000),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        orderVec = np.random.permutation(len(self.transforms))
        for t in orderVec:
            data = self.transforms[t](data)
        return data

class Lambda(object):
    """
    Apply a user-defined lambda as a transform.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

class Scale(object):
    """
    Scale a tensor from a 16-bit integer (represented as a FloatTensor)
    to a floating point number between -1.0 and 1.0.  Note the 16-bit number is
    called the "bit depth" or "precision", not to be confused with "bit rate".
    Args:
        factor (int): maximum value of input tensor. default: 16-bit depth
    """

    def __init__(self, factor=2**31):
        self.factor = factor

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of audio of size (Samples x Channels)
        Returns:
            Tensor: Scaled by the scale factor. (default between -1.0 and 1.0)
        """
        if isinstance(tensor, (torch.LongTensor, torch.IntTensor)):
            tensor = tensor.float()
        return tensor / self.factor
'''
class ToPytorch(object):
    """
    Convert ndarrays to Pytorch Tensors.
    """

    def __init__(self, transpose=False):
        self.transpose = transpose;

    def __call__(self, data):
        if (self.transpose):
            data = data.transpose((2, 0, 1))
        return torch.from_numpy(data)

class ToTensorflow(object):
    """
    Convert ndarrays to Pytorch Tensors.
    """

    def __init__(self, transpose=False):
        self.transpose = transpose;

    def __call__(self, data):
        if (self.transpose):
            data = data.transpose((2, 0, 1))
        data = np.asarray(data, np.float32)
        return tf.convert_to_tensor(data, np.float32)
 '''  