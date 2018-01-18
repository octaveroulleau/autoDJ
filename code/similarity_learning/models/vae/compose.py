""" Running mode.
"Random walk" in the VAE embedding space (with small cumulated distance)
Returns a list of mixing points : that can be used to re-synthetize a new track
1. Define constraints for composition : number of chunks, probability of switching track
2. Access the embedding space and perform a random walk with pre-defined constraints : track matching
3. From the chunk’s labels returned, create a list of mixing points
"""

# Random walk : draw a line in latent space, discretize, find nearest neighbors.