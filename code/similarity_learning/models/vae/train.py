""" Training mode.
1. Get the dataflow from the CNN’s latent space (1 chunk = 1 point + 1 label)
2. Build VAE base architecture
3. Train the VAE unsupervised on this data (1 track = 1 valid path)
4. Adjust the parameters to get a better embedding space
5. Freeze the model (save)
"""


# VAE model

# VAE training
# Task : auto-encoding of the chunks' features (sort of t-sne : organise space), dim 10.
# Alternatively, concatenate 3 chunks and organise a "track space"

def save_vae_model(model, filename):
  """ Saves trained VAE, serialized, to a file for reuse (evaluation and run)
  """