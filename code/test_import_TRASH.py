import data
from similarity_learning.models.asynchronous.asynchronous import asynchronous_learning
#%%
audioSet, audioOptions = data.import_data.import_data()
#%%
transform_type, transform_options = audioSet.getTransforms()
asynchronous_learning(audioSet, audioOptions)
