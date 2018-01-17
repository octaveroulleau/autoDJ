import data
from similarity_learning.models.asynchronous.asynchronous import asynchronous_learning
#%%
audioSet, audioOptions = data.import_data.import_data()
#%%
transform_type, transform_options = audioSet.getTransforms()
audioSet.files = audioSet.files[1:21]
nb_frames = 1000

asynchronous_learning(audioSet, audioOptions, nb_frames, model_type, model_options)
