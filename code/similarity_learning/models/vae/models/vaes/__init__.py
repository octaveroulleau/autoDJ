# -*-coding:utf-8 -*-
 
"""
    The ``vaes`` module
    ========================
 
    This package defines different models
 
    Examples
    --------
 
    Subpackages available
    ---------------------
 
    Comments and issues
    ------------------------
    None for the moment
 
    Contributors
    ------------------------
    * Philippe Esling       (esling@ircam.fr)
 
"""
 
# info
__version__ = "1.0"
__author__  = "chemla@ircam.fr"
__date__    = ""
__all__     = []
 
# import sub modules
from .VanillaVAE import VanillaVAE
from .VanillaDLGM import VanillaDLGM
from .ConditionalVAE import ConditionalVAE
#from .SSVAE import SSVAE
from .variational_modules import VariationalLayer
