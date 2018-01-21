# -*-coding:utf-8 -*-
 
"""
    The ``transforms`` module
    ========================
 
    This package allows to perform data augmentations based on different types
    Currently it contains five implementations
        * Generic       : Generic augment operation (Composition, Random)
        * Matrix        : Matrix type augmentations
        * Signal        : 1d signal operations
        * Spectral      : Augmentations over spectral transforms
        * Symbolic      : Augmentations over symbolic scores
 
    Subpackages available
    ---------------------

        * Generic
        * Matrix
        * Signal
        * Spectral
        * Symbolic
 
    Comments and issues
    ------------------------
        
        None for the moment
 
    Contributors
    ------------------------
        
    * Philippe Esling       (esling@ircam.fr)
 
"""
 
# info
__version__ = "1.0"
__author__  = "esling@ircam.fr"
__date__    = ""
__all__     = ["generic", "matrix", "spectral"]
 
# import sub modules
from . import generic
from . import matrix
from . import spectral