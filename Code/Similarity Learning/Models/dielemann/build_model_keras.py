# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 18:59:58 2017

@author: octav
"""

from keras.models import Sequential, Model
import keras.layers as layers
import keras.backend as K
import pdb
import numpy as np
#%%

def build_full_model(frames, 
                     freq_bins, 
                     mod_options):
    '''
    Build a keras model based on the Dielemann model for music recommandation.
    This model is trained on time-frequency representation of music. The input 
    data fed to the model must be shaped like (batch_size, time, frequency).
    
    Parameters
    ----------
    frames: int
        Number of time frames in a single input
        
    freq_bin: int
        Number of frequency bands in a single input
        
    mod_option: dictionnary
        Specific options for the model. This dictionnary must contain:
            
        'activation': string
            name of the activation function for keras layers
            
        'batchNormConv': bool
            Choose if the model should apply babtch normalization after each 
            convnet
            
        'FC number': int
            Number of cells for each FC layer
            
        'batchNormDense': bool
            Choose if the model should apply babtch normalization after each 
            FC layer
        
        'Alphabet size': int
            Size of the training alphabet (last layer, size of the output)
        
    Returns
    -------
    model: keras model
        The model built. Expects inputs of shape (batch_size, frames, 
        freq_bins) and outputs tensor of shape (batch_size, alphabet_size).
    '''
    
    inputs = layers.Input(shape = (frames, freq_bins))
    
    #%%=========== First layer ===============
    #zero-padding the input
    padding_1 = layers.ZeroPadding1D(padding = 2)(inputs)
    #Convnet 256 neurons with 4 sample window. Activation defined in mod_option dictionnary
    conv1 = layers.Conv1D(256, 4, padding = 'same', activation = mod_options['activation'])(padding_1)
    #Normalise batch if defined in mod_options
    if mod_options['batchNormConv']:
        conv1 = layers.BatchNormalization()(conv1)
    #Reduce data by max pooling between 2 values
    pool_1 = layers.MaxPooling1D(pool_size = 2)(conv1)
    
    
    #%%============ Second layer ==============
    #Same layer as the previous one
    padding_2 = layers.ZeroPadding1D(padding = 2)(pool_1)
    conv_2 = layers.Conv1D(256, 4, padding = 'same', activation = mod_options['activation'])(padding_2)
    if mod_options['batchNormConv']:
        conv_2 = layers.BatchNormalization()(conv_2)
    pool_2 = layers.MaxPooling1D(pool_size = 2)(conv_2)

    
    '''
    #%%=========== Third layer ???================
    #zero-padding the input
    model.Add(layers.ZeroPadding1D(padding = 2))
    #Convnet 512 neurons with 4 sample window.
    model.add(layers.Conv1D(512, 4, padding = 'same', activation = mod_options['activation']))
    #Normalize batch if defined
    if mod_options['batchNormConv']:
        model.add(layers.BatchNormalization())
    
    '''

    
    #%%=========== Fourth layer =================
    #zero-padding the input
    padding_3 = layers.ZeroPadding1D(padding = 2)(pool_2)
    #Convnet 512 neurons with 4 sample window.
    conv_3 = layers.Conv1D(512, 4, padding = 'same', activation = mod_options['activation'])(padding_3)
    #Normalize batch if defined
    if mod_options['batchNormConv']:
        conv_3 = layers.BatchNormalization()(conv_3)
        
   
    
    #%%========== Global temporal pooling layer =========
    pdb.set_trace()
    pool_max = layers.GlobalMaxPooling1D()(conv_3)
    pool_average = layers.GlobalAveragePooling1D()(conv_3)
    pool_LP = layers.Lambda(lambda x:  GlobalLPPooling1D(x))(conv_3)
    
    pool_time = layers.Concatenate()([pool_max, pool_average, pool_LP])
    
    #%%========== FC Layers =========================
    FC_1 = layers.Dense(mod_options['FC number'], activation = mod_options['activation'])(pool_time)
    if mod_options['batchNormDense']:
        FC_1 = layers.BatchNormalization()(FC_1)
        
    FC_2 = layers.Dense(mod_options['FC number'], activation = mod_options['activation'])(FC_1)
    if mod_options['batchNormDense']:
        FC_2 = layers.BatchNormalization()(FC_2)
        
    FC_3 = layers.Dense(mod_options['Alphabet size'], activation = 'softmax')(FC_2)
    
    model = Model(inputs = inputs, outputs = FC_3)
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model
    
#%%
def build_conv_layers(frames, freq_bins, mod_options):
    '''
    Build the convolutionnal base of the model based on the Dielemann model 
    for music recommandation (useful for transfer learning).
    This model is trained on time-frequency representation of music. The input 
    data fed to the model must be shaped like (batch_size, time, frequency).
    Once FC layers are added via the add_fc_layers function, the model can be 
    trained and saved for re-use in transfer learning applications.
    
    Parameters
    ----------
    frames: int
        Number of time frames in a single input
        
    freq_bin: int
        Number of frequency bands in a single input
        
    mod_option: dictionnary
        Specific options for the model. This dictionnary must contain:
            
        'activation': string
            name of the activation function for keras layers
            
        'batchNormConv': bool
            Choose if the model should apply babtch normalization after each 
            convnet
    Returns
    -------
    model: keras model
        The model is not compiled since it requires FC layers on top. Expects 
        inputs of shape (batch size, frames, freq_bins) and outputs tensor of 
        shape: (batch size, frames_pool, freq_bins) where frames_pool has been 
        through 3 poolings of size 2 (divise by 2) and with 4 zeros added 
        before each pooling (add 4).
    
    '''
    
    inputs = layers.Input(shape = (frames, freq_bins))
    
    #%%=========== First layer ===============
    #zero-padding the input
    padding_1 = layers.ZeroPadding1D(padding = 2)(inputs)
    #Convnet 256 neurons with 4 sample window. Activation defined in mod_option dictionnary
    conv1 = layers.Conv1D(256, 4, padding = 'same', activation = mod_options['activation'])(padding_1)
    #Normalise batch if defined in mod_options
    if mod_options['batchNormConv']:
        conv1 = layers.BatchNormalization()(conv1)
    #Reduce data by max pooling between 2 values
    pool_1 = layers.MaxPooling1D(pool_size = 2)(conv1)
    
    #%%============ Second layer ==============
    #Same layer as the previous one
    padding_2 = layers.ZeroPadding1D(padding = 2)(pool_1)
    conv_2 = layers.Conv1D(256, 4, padding = 'same', activation = mod_options['activation'])(padding_2)
    if mod_options['batchNormConv']:
        conv_2 = layers.BatchNormalization()(conv_2)
    pool_2 = layers.MaxPooling1D(pool_size = 2)(conv_2)
    
    '''
    #%%=========== Third layer ???================
    #zero-padding the input
    model.Add(layers.ZeroPadding1D(padding = 2))
    #Convnet 512 neurons with 4 sample window.
    model.add(layers.Conv1D(512, 4, padding = 'same', activation = mod_options['activation']))
    #Normalize batch if defined
    if mod_options['batchNormConv']:
        model.add(layers.BatchNormalization())
    
    '''
    
    #%%=========== Fourth layer =================
    #zero-padding the input
    padding_3 = layers.ZeroPadding1D(padding = 2)(pool_2)
    #Convnet 512 neurons with 4 sample window.
    conv_3 = layers.Conv1D(512, 4, padding = 'same', activation = mod_options['activation'])(padding_3)
    #Normalize batch if defined
    if mod_options['batchNormConv']:
        conv_3 = layers.BatchNormalization()(conv_3)
    
    model = Model(inputs = inputs, outputs = conv_3)
    
    return model


#%%
    
def add_fc_layers(base_model, mod_options):
    '''
    Add FC layers to a previously constructed base model (useful for transfer 
    learning). The added layers are based on the construction of the Dielemann 
    model for music recommendation, starting from a base model of 
    convolutionnal layers.
    
    Parameters
    ----------
    base_model: keras model
        The model that we wish to stack layers upon. 
    
    mod_options: dictionnary
        Specific options for the model. This dictionnary must contain:
            
        'FC number': int
            Number of cells for each FC layer
            
        'batchNormDense': bool
            Choose if the model should apply babtch normalization after each 
            FC layer
        
        'Alphabet size': int
            Size of the training alphabet (last layer, size of the output)
            
        'Freeze layer': bool
            Choose if you want to freeze the convolutionnal layers state for 
            training (only the FC layers will be updated during training if 
            set to True)
            
    Returns
    -------
    model: keras model
        The output model. Expects inputs of shape as defined in the base_model 
        and outputs a tensor of shape (batch size, alphabet size).
    '''
    
    inputs = base_model.output
    
    #%%========== Global temporal pooling layer =========
    pool_max = layers.GlobalMaxPooling1D()(inputs)
    pool_average = layers.GlobalAveragePooling1D()(inputs)
    pool_LP = layers.Lambda(lambda x:  GlobalLPPooling1D(x))(inputs)
    
    pool_time = layers.Concatenate()([pool_max, pool_average, pool_LP])
    
    #%%========== FC Layers =========================
    FC_1 = layers.Dense(mod_options['FC number'], activation = mod_options['activation'])(pool_time)
    if mod_options['batchNormDense']:
        FC_1 = layers.BatchNormalization()(FC_1)
        
    FC_2 = layers.Dense(mod_options['FC number'], activation = mod_options['activation'])(FC_1)
    if mod_options['batchNormDense']:
        FC_2 = layers.BatchNormalization()(FC_2)
        
    FC_3 = layers.Dense(mod_options['Alphabet size'], activation = 'softmax')(FC_2)
    

    
    model = Model(inputs = base_model.input, outputs = FC_3)
    
    #%%========= Freeze Convnets if necessary
    if mod_options['Freeze layer']:
        for layer in base_model.layers:
            layer.trainable = False
            
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model
    
#%%

def GlobalLPPooling1D(x, p = 2):
    '''
    Perform a global pooling on 1-dimensionnal data (global temporal pooling)
    using a L-p norm.
    
    Parameters
    ----------
    x: keras tensor
        Tensor to pool across. The function will pool across dimension 1
    p: int
        Norm class. Default is 2
    
    Returns
    -------
    x_pool: keras tensor
        A keras tensor containing all L-p norms of the input across dimension 1
        If x is shaped as (n1,n2,n3,etc), x_pool will be (n1,n3,etc)
    '''
    x_pool = K.pow(x,p)
    x_pool = K.sum(x_pool,axis = 1)
    x_pool = K.sqrt(x_pool)
    return x_pool
  
#%% Test section

mod_options = {
        'activation': 'relu',
        'batchNormConv': True,
        'FC number': 2048,
        'batchNormDense': True,
        'Alphabet size': 10,
        'Freeze layer': True}

base_model = build_conv_layers(599, 128, mod_options)

model = add_fc_layers(base_model, mod_options)
