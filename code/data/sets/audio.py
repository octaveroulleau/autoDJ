"""

 Import toolbox       : Audio dataset import

 This file contains the definition of an audio dataset import.

 Author               : Philippe Esling
                        <esling@ircam.fr>

"""
#from matplotlib import pyplot as plt
import pdb
import numpy as np
import scipy as sp
import os
# Package-specific import
from . import generic as dat
from . import utils as du
import time

"""
###################################
# Initialization functions
###################################
"""

class DatasetAudio(dat.Dataset):
    """ Definition of a basic dataset
    Attributes:
        dataDirectory:
    """

    def __init__(self, options):
        super(DatasetAudio, self).__init__(options)
        # Accepted types of files
        self.types = options.get("types") or ['mp3', 'wav', 'wave', 'aif', 'aiff', 'au'];
        self.importBatchSize = options.get("importBatchSize") or 64;
        self.transformType = options.get("transformType") or ['stft'];
        self.matlabCommand = options.get("matlabCommand") or '/Applications/MATLAB_R2015b.app/bin/matlab';
        self.forceRecompute = options.get("forceRecompute") or False;
        # Type of audio-related augmentations
        self.augmentationCallbacks = [];

    """
    ###################################
    # Import functions
    ###################################
    """

    def importData(self, idList, options):
        """ Import part of the audio dataset (linear fashion) """
        options["matlabCommand"] = options.get("matlabCommand") or self.matlabCommand;
        options["transformType"] = options.get("transformType") or self.transformType;
        options["dataDirectory"] = options.get("dataDirectory") or self.dataDirectory;
        options["analysisDirectory"] = options.get("analysisDirectory") or self.analysisDirectory;
        options["forceRecompute"] = options.get("forceRecompute") or self.forceRecompute;
        options["verbose"] = options.get("verbose") or self.verbose;
        # We will create batches of data
        indices = []
        # If no idList is given then import all !
        if (idList is None):
            indices = np.linspace(0, len(self.files) - 1, len(self.files))
            indices = np.pad(indices, (0, self.importBatchSize - (len(self.files) % self.importBatchSize)), 'constant', constant_values=0)
            indices = np.split(indices, len(indices) / self.importBatchSize)
        else:
            indices = np.array(idList)
        if (len(self.data) == 0):
            self.data = [None] * len(self.files)
        # Parse through the set of batches
        for v in indices:
            curFiles = [None] * v.shape[0]
            for f in range(v.shape[0]):
                curFiles[f] = self.files[int(v[f])]
            curData, curMeta = importAudioData(curFiles, options)
            for f in range(v.shape[0]):
                self.data[int(v[f])] = curData[f]

    """
    ###################################
    # Get asynchronous pointer and options to import
    ###################################
    """
    def getAsynchronousImport(self):
        a, transformOpt = self.getTransforms()
        options = {
            "matlabCommand":self.matlabCommand,
            "transformType":self.transformType,
            "dataDirectory":self.dataDirectory,
            "analysisDirectory":self.analysisDirectory,
            "forceRecompute":self.forceRecompute,
            "transformOptions":transformOpt,
            "verbose":self.verbose
            }
        return importAudioData, options

    def windowData(self, target_size, overlap, options):
        new_data = []
        new_metadata = []
        # Temporarily separate if phase is concatenated
        new_size = target_size * self.data[0].shape[1]
        for task, v in pairs(self.metadata):
            new_metadata[task] = {};
        for i, data in ipairs(self.data):
            current_size = data.shape[0] # number of bins
            if current_size < target_size:
                return None
            # if concatenation, split data in two
            if (options.concatenatePhase == 1):
                data = data.split(current_size / 2)
                current_size = current_size / 2
            # total number of windows
            final_size = np.floor((current_size - target_size) / (target_size - overlap))
            for j in range(final_size):
                idx_beg = j * (target_size - overlap)
                idx_end = (j + 1) * target_size - j * overlap
                if (options.concatenatePhase == 1):
                    new_data[len(new_data) + 1] = nn.JoinTable(1).forward(data[1].narrow(1, idx_beg, target_size), data[2].narrow(1, idx_beg, target_size))
                else:
                    new_data[len(new_data) + 1] = data.narrow(1, idx_beg, target_size)
                for task, metadata in pairs(self.metadata):
                    new_metadata[task][len(new_metadata[task]) + 1] = metadata[i]
        self.data = None
        self.metadata = None
        self.data = new_data
        self.metadata = new_metadata

    """
    ###################################
    # Obtaining transform set and options
    ###################################
    """

    def getTransforms(self):
        """
        Transforms (and corresponding options) available
        """
        # List of available transforms
        transformList = [
            'stft',               # Short-Term Fourier Transform
            'mel',                # Log-amplitude Mel spectrogram
            'mfcc',               # Mel-Frequency Cepstral Coefficient
            'gabor',              # Gabor features
            'chroma',             # Chromagram
            'cqt',                # Constant-Q Transform
            'gammatone',          # Gammatone spectrum
            'dct',                # Discrete Cosine Transform
            'hartley',            # Hartley transform
            'rasta',              # Rasta features
            'plp',                # PLP features
#            'wavelet',            # Wavelet transform
#            'scattering',         # Scattering transform
            'cochleogram',        # Cochleogram
            'strf',               # Spectro-Temporal Receptive Fields
            'modulation'          # Modulation spectrum
        ];
        # List of options
        transformOptions = {
            "debugMode":0,
            "resampleTo":22050,
            "targetDuration":0,
            "winSize":2048,
            "hopSize":1024,
            "nFFT":1024,
            # Normalization
            "normalizeInput":0,
            "normalizeOutput":0,
            "equalizeHistogram":0,
            "logAmplitude":1,
            # Phase
            "removePhase":1,
            "concatenatePhase":0,
            # Mel-spectrogram
            "minFreq":64,
            "maxFreq":8000,
            "nbBands":128,
            # Mfcc
            "nbCoeffs":13,
            "delta":0,
            "dDelta":0,
            # Gabor features
            "omegaMax":'[pi/2, pi/2]',
            "sizeMax":'[3*nbBands, 40]',
            "nu":'[3.5, 3.5]',
            "filterDistance":'[0.3, 0.2]',
            "filterPhases":'{[0, 0], [0, pi/2], [pi/2, 0], [pi/2, pi/2]}',
            # Chroma
            "chromaWinSize":2048,
            # CQT
            "cqtBins":24,
            "cqtFreqMin":64,
            "cqtFreqMax":8000,
            "cqtGamma":0.5,
            # Gammatone
            "gammatoneBins":64,
            "gammatoneMin":64,
            "gammatoneMax":8000,
            # Wavelet
            "waveletType":'\'gabor_1d\'',
            "waveletQ":8,
            # Scattering
            "scatteringDefault":1,
            "scatteringTypes":'{\'gabor_1d\', \'morlet_1d\', \'morlet_1d\'}',
            "scatteringQ":'[8, 2, 1]',
            "scatteringT":8192,
            # Cochleogram
            "cochleogramFrame":64,        # Frame length, typically, 8, 16 or 2^[natural #] ms.
            "cochleogramTC":16,           # Time const. (4, 16, or 64 ms), if tc == 0, the leaky integration turns to short-term avg.
            "cochleogramFac":-1,          # Nonlinear factor (typically, .1 with [0 full compression] and [-1 half-wave rectifier]
            "cochleogramShift":0,         # Shifted by # of octave, e.g., 0 for 16k, -1 for 8k,
            "cochleogramFilter":'\'p\'',      # Filter type ('p' = Powen's IIR, 'p_o':steeper group delay)
            # STRF
            "strfFullT":0,                # Fullness of temporal margin in [0, 1].
            "strfFullX":0,                # Fullness of spectral margin in [0, 1].
            "strfBP":0,                   # Pure Band-Pass indicator
            "strfRv":'2 .^ (1:.5:5)',     # rv: rate vector in Hz, e.g., 2.^(1:.5:5).
            "strfSv":'2 .^ (-2:.5:3)',    # scale vector in cyc/oct, e.g., 2.^(-2:.5:3).
            "strfMean":0                  # Only produce the mean activations
        }
        return transformList, transformOptions;

    def __dir__(self):
        tmpList = super(DatasetAudio, self).__dir__()
        return tmpList + ['importBatchSize', 'transformType', 'matlabCommand']
    '''
    def plotExampleSet(self, setData, labels, task, ids):
        fig = plt.figure(figsize=(12, 24))
        ratios = np.ones(len(ids))
        fig.subplots(nrows=len(ids),ncols=1,gridspec_kw={'width_ratios':[1], 'height_ratios':ratios})
        for ind1 in range(len(ids)):
            ax = plt.subplot(len(ids), 1, ind1 + 1)
            if (setData[ids[ind1]].ndim == 2):
                ax.imshow(np.flipud(setData[ids[ind1]]), interpolation='nearest', aspect='auto')
            else:
                tmpData = setData[ids[ind1]]
                for i in range(setData[ids[ind1]].ndim - 2):
                    tmpData = np.mean(tmpData, axis=0)
                ax.imshow(np.flipud(tmpData), interpolation='nearest', aspect='auto')
            plt.title('Label : ' + str(labels[task][ids[ind1]]))
            ax.set_adjustable('box-forced')
        fig.tight_layout()

    def plotRandomBatch(self, task="genre", nbExamples=5):
        setIDs = np.random.randint(0, len(self.data), nbExamples)
        self.plotExampleSet(self.data, self.metadata, task, setIDs)
    '''
"""
###################################
# External functions for audio import
###################################
"""
def importAudioData(curBatch, options):
    # Prepare the call to matlab command
    finalCommand = options["matlabCommand"] + ' -nodesktop -nodisplay -nojvm -r '
    # Add the transform types
    transformString = "{";
    for it in range(len(options['transformType'])):
        transformString = transformString + '\'' + options["transformType"][it] + ((it < (len(options["transformType"]) - 1)) and '\',' or '\'}')
        print(transformString)
        pause(0.5)
        
    finalCommand = finalCommand + ' "transformType=' + transformString + '; oldRoot = \'' + options["dataDirectory"] +  '\'; newRoot = \'' + options["analysisDirectory"] + '\'';
    # Find the path of the current toolbox (w.r.t the launching path)
    curPath = os.path.realpath(__file__)
    curPath = os.path.split(curPath)[0]
    # Now handle the eventual options
    if (options["transformOptions"]) and (not (options.get("defaultOptions") == True)):
        for t, v in options["transformOptions"].items():
            finalCommand = finalCommand + '; ' + t + ' = ' + str(v)
    finalData = [None] * len(curBatch)
    finalMeta = [None] * len(curBatch)
    # Parse through the set of batches
    curAnalysisFiles = [None] * len(curBatch)
    audioList = [None] * len(curBatch)
    curIDf = 0
    # Check which files need to be computed
    for i in range(len(curBatch)):
        curFile = curBatch[i]
        curAnalysisFiles[i] = os.path.splitext(curFile.replace(du.esc(options["dataDirectory"]), options["analysisDirectory"]))[0] + '.npy'
        try:
            fIDTest = open(curAnalysisFiles[i], 'r')
        except IOError:
            fIDTest = None
        if ((fIDTest is None) or (options["forceRecompute"] == True)):
            audioList[curIDf] = curFile
            curIDf = curIDf + 1
        else:
            fIDTest.close()
    audioList = audioList[:curIDf]
    # Some of the files have not been computed yet
    if (len(audioList) > 0):
        unprocessedString = ""
        if options["verbose"]:
            print("* Computing transforms ...")
        # Matlab processing for un-analyzed files
        for f in range(len(audioList)):
            audioList[f] =  audioList[f].replace("'","''")
            unprocessedString = unprocessedString + '\'' + audioList[f] + (f < (len(audioList) - 1) and '\',' or '\'')
            audioList[f] =  audioList[f].replace("''","'")
        if options["verbose"]:
            print(str(len(audioList)) + ' analysis files not found.')
            print("Launching Matlab...")
        curCommand = finalCommand + '; audioList = {' + unprocessedString + '}; cd ' + curPath + '/cortical-audio; processSound; exit;"'
        print(curCommand)
        fileN = os.popen(curCommand)
        output = fileN.read();
        if options["verbose"]:
            print(output)
        fileN.close()
        for f in range(len(audioList)):
            curFile = os.path.splitext(audioList[f].replace(du.esc(options["dataDirectory"]), options["analysisDirectory"]))[0]
            print(curFile)
            try:
                curData = sp.io.loadmat(curFile + '.mat');
                # TODO : Here I save just a single transform
                curData = curData["transforms"][options["transformType"][0]][0][0]
                np.save(curFile + '.npy', curData);
                os.popen('rm ' + curFile.replace(' ', '\\ ') + '.mat');
            except:
                pass
    for f in range(len(curAnalysisFiles)):
        curData = np.load(curAnalysisFiles[f]);
        finalData[f] = curData;
        finalMeta[f] = 0;
    return finalData, finalMeta;
