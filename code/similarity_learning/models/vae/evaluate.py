""" Evaluation mode.
1. Defnition of a custom score : distance between two consecutive chunks of a track should be minimal
2. Evaluate score on the training dataset
3. Use T-SNE to visually check coherence of the embedding space
"""

# Use toymix !!! (chunk by chunk)

def forward(cnn_data, model):
	""" Input training set in CNN's feature space
	Output the same set in the VAE's latent space
	"""

def score_vae(model):
	"""
	For each track, find the path representing the chunks it is composed of in the latent space
	Cumulate length of all paths
	Print score of the model
	"""

def t_sne_train(model):
	"""
	Plots a representation of the training data in the latent space using t-sne dimensionality reduction.
	Plots each chunk along with its label.
	Plots tracks as paths.
	"""





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


def test_VAE_PCA(self):
    # try to load a vae
    vaeLoaded = loadVAE('dummyDataset98_NPZ_E<1024-relu6-401-muSig-6>_D<6-relu6-399-sigmoid-1024>_beta1_mb49_lr0dot001_ep11','./dummySaveTest/')
    test = numpy.load("./dummyDataset/dummyDataset98.npz")
    # vaeLoaded = loadVAE('toy-spectral-richness-v2-lin_NPZ_E<1024-relu6-600-muSig-10>_D<10-relu6-600-muSig-1024>_beta3_mb100_lr0dot001_ep400',
    #                     "../data/savedVAE/Alexis-nonNorm/beta3/WU100/")
    # test = numpy.load("../data/toy-spectral-richness-v2-lin.npz")
    spec = test['Spectrums']
    lbl = test['labels']

    lbltest = []
    lbltest = lbl[:, 0:98:]
    spec = spec[:, 0:98]
    print("~~~ Nb samples :", len(lbltest[0, :]))

    spec = torch.from_numpy(spec)
    spec = spec.float()
    spec = spec.t()
    lblarr = numpy.array(lbltest)

    spec = Variable(spec)

    z_res = vaeLoaded.encode(spec)
    z_res = z_res.data
    z_res = z_res.double()

    PCA_vision(z_res, lblarr, 0)
    plt.figure()

    z_res = z_res.numpy()

    Y = tsne(z_res, 2, 10)
    plt.scatter(Y[:, 0], Y[:, 1], 20, lblarr[0])
    plt.show()

    self.assertTrue(vaeLoaded.loaded)
