#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:46:21 2018

@author: chemla
"""

from numpy import zeros

def OneHot(labels, dim):
    n = labels.shape[0]
    t = zeros((n, dim))
    for i in range(n):
        t[i, labels[i]] = 1
    return t