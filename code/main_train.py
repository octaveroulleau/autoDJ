# Here the file to run to execute the full process in train mode
import torch
import tensorflow as tf
import keras
import data
from similarity_learning.models.asynchronous.asynchronous import asynchronous_learning
from keras.backend.tensorflow_backend import set_session
#%% GPU configuration: check GPU number for training
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.visible_device_list = "2"
set_session(tf.Session(config=config))

# Load data

# Pre-process data

# Train CNNs asynchronously + save model

# Train VAE + save model
