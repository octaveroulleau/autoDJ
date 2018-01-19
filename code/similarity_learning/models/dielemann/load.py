# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:43:53 2018

@author: octav
"""

from keras.models import load_model
import pickle

#%%
def load_CNN_model(model_name, base_dir = './similarity_learning/models/dielemann/models/', model_type = 'base'):
    filepath_model = base_dir + model_name + '_'+model_type+'.h5'
    filepath_options = base_dir + model_name + '_options'
    
    model = load_model(filepath_model)
    model_options = pickle.load(open(filepath_options,'rb'))
    
    print(model_name+'_'+model_type+' loaded')
    return model, model_options
    
    