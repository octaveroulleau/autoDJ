# Code Documentation

## Files & folders
`src/` contains all Modules to create, train, save, load, visualize **Variational autoencoder** (VAE) with the wanted dataset.

   > * **mainScript.py** is the script used either to train a VAE with a `.npz` dataset (see **README.md** in `../toy`) or load an already trained VAE and do some things like reconstruction analysis, latent space visualization, generation...\
>* Creation, training, saving and loading are handled by **VAE.py** and **EncoderDecoder.py**.\
>* **ManageDataset.py** in `dataset/` makes the link between the training dataset and the VAE.
Once the VAE is trained, `Visualize/` folder gives tools and functions to visualize and analyse it.

`data/` is the default folder where trained VAEs are stored\

`unitTest/` contains helpful simple tests on VAE classes, datasets and analysis to make sure  everything is ok when source code is changed
>* **EncoderDecoderTest.py** tests Encoder/Decoder structure
>* **DatasetTest.py** tests dataset handling
>* **VAETest.py** tests VAE train, save, load, analyse, visualize

### Warnings
Before each commit **run unit tests** in `./unitTest/` folder

Name of datasets **should NOT** contain any '_' characters

**mb-size** needs to be a divider of total dataset length.

### Use of **mainScript.py**

### Help
```{r, engine='bash', count_lines}
cd src/
```

```{r, engine='bash', count_lines}
python mainScript.py --help
```

## 1. Training

```{r, engine='bash', count_lines}
cd src/
```

#### Immediate test
To train a VAE  on a dummy dataset of filepath `../data/dummyDataset98.npz`. It's composed of 100 spectra of length 1024. 
The command:
```{r, engine='bash', count_lines}
python mainScript.py -encoderIOdims 1024 600 10 -decoderIOdims 10 600 1024 -encoderNL "relu6" -decoderNL "relu6" -mb-size 49 -dataKey "Spectrums"
```
will by default load this dataset. At the end of the training, the VAE is saved into the default save path `../data/dummySave`
Bernoulli equivalent :
```{r, engine='bash', count_lines}
python mainScript.py -encoderIOdims 1024 600 10 -decoderIOdims 10 600 1024 -encoderNL "relu6" -decoderNL "relu6" "sigmoid" -type "bernoulli" -mb-size 49 -dataKey "Spectrums"
```

#### Example
To train a **Gaussian VAE** on "dataset.npz" dataset of 44800 data of dimension 1024, whose latent dimensions is 10, 1-'relu'-layer NN for Encoder and Decoder and save it in `../data/savedVAE/` folder, the total command should be:
```{r, engine='bash', count_lines}
python mainScript.py -mode "train" -encoderIOdims 1024 600 10 -decoderIOdims 10 600 1024 -encoderNL "relu6" -decoderNL "relu6" -type "gaussian" -dataset-path "../data/dataset.npz" -dataKey "images" -save-path "../data/savedVAE/" -mb-size 10 -epochs 10  
```
The command can be reduced as it has default values:

```{r, engine='bash', count_lines}
python mainScript.py -encoderIOdims 1024 600 10 -decoderIOdims 10 600 1024 -encoderNL "relu6" -decoderNL "relu6" -dataset-path "../data/dataset.npz" -save-path "../data/savedVAE/" 
```
 For a **Bernoulli VAE**, the equivalent command will be : 

```{r, engine='bash', count_lines}
python mainScript.py -encoderIOdims 1024 600 10 -decoderIOdims 10 600 1024 -encoderNL "relu6" -decoderNL "relu6" "sigmoid" -type "bernoulli" -dataset-path "../data/dataset.npz" -save-path "../data/savedVAE/" 
```
**More flags**
- Warm-up (in number of epochs)
```{r, engine='bash', count_lines}
-Nwu 100
```
- Noise input during training (gain value of a random gaussian noise)
```{r, engine='bash', count_lines}
-noise 1.5
```


## 2. Loading VAE, view & generation

Instead of using the default mode **"train"** in command, use the mode **"load"** with flag. This mode enables to use trained VAE and do some stuff with it (e.g. PCA, t-sne, generation ...).
```{r, engine='bash', count_lines}
-mode "load"
```


## 3. Unit Testing
```{r, engine='bash', count_lines}
cd unitTest/
```

* Tests on dataset handler
```{r, engine='bash', count_lines}
python DatasetTest.py
```
* Tests on encoder/decoder structure
```{r, engine='bash', count_lines}
python EncoderDecoderTest.py
```
* Various tests on VAE (learning, saving, loading, visualize)
```{r, engine='bash', count_lines}
python VAETest.py
```









