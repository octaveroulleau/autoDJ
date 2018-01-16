import data
from similarity_learning.models.asynchronous.asynchronous import asynchronous_learning
#%%
audioSet, audioOptions = data.import_data.import_data()
#%%
asynchronous_learning(audioSet, audioOptions)
