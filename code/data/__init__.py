# -*-coding:utf-8 -*-

"""
    The ``data`` module
    ========================

    This package allows to perform all data-related functions.
    Currently it contains two sub-packages
        * Sets          : Data containers (datasets)
        * Transforms    : Data augmentations

    :Example:

    >>> from data.sets import DatasetAudio
    >>> DatasetAudio()

    Subpackages available
    ---------------------

        * Sets
        * Transforms

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
__all__     = ["sets", "transforms"]

# import sub modules
from . import sets
from . import transforms
from . import import_data
