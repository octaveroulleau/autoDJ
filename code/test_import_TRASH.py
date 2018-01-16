import data
from similarity_learning.models.asynchronous.asynchronous import asynchronous_learning
#%%
audioSet = data.import_data.import_data()
#%%
asynchronous_learning(audioSet)
