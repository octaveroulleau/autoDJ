""" Running mode.
"Random walk" in the VAE embedding space (with small cumulated distance)
Returns a list of mixing points : that can be used to re-synthetize a new track
1. Define constraints for composition : number of chunks, probability of switching track
2. Access the embedding space and perform a random walk with pre-defined constraints : track matching
3. From the chunk’s labels returned, create a list of mixing points
"""

# Random walk : draw a line in latent space, discretize, find nearest neighbors.

# get savefile path
directory = args.vae_path
savefile = directory.split("/")[-1]
directory = directory.replace(savefile, "")
# load vae
vaeLoaded = loadVAE(savefile, directory)

def test_VAE_load(self):
    # try to load a vae 
    vae = loadVAE('dummyDataset98_NPZ_E<1024-relu6-600-muSig-10>_D<10-relu6-600-muSig-1024>_beta1_mb49_lr0dot001_ep5',
                  './dummySaveTest/')
    self.assertTrue(vae.created and vae.loaded)