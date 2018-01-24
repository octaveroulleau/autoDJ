# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 18:06:21 2018

@author: octav
"""
# import sys
# sys.path.append('similarity_learning/')
from similarity_learning.models.asynchronous.task import AsynchronousTask
import similarity_learning.models.dielemann.build as build
import numpy as np
import keras
import pickle
from pre_processing.chunkify import track_to_chunks
import skimage.transform as skt
from similarity_learning.models.dielemann.build import GlobalLPPooling1D
import pdb

def chunk_loader(idx, 
                 dataIn, 
                 options):
    '''
    task pointer used to load tracks and split them to chunks asynchronously
    
    Parameters
    ----------
    idx:
        Track index
    dataIn:
        Track filepath
    options:
        Dictionnary of options. Must contains:
            -audioSet: object of type DatasetAudio
            -frames number: int, defines target resampling frames number
            -task: string, defines metadata associated with each chunk (artist, genre etc...)
            -AudioOptions from dataset
    
    Returns
    -------
    data: list of numpy arrays containing cqt of each chunk
    meta: list of string containing metadata informations for each chunk
    '''

    audioSet = options['audioSet']

    downbeat = audioSet.metadata["downbeat"][idx][0]
    Fs = 44100
    chunks = track_to_chunks(idx, Fs, downbeat)

    data = []
    meta = []
    
    for i in range(len(chunks)):
        chunk = chunks[i].get_cqt(audioSet, options)
        nbBins = chunk.shape[0]
        chunk = skt.resize(chunk, (nbBins, options["frames number"]), mode='reflect')
        data.append(chunk)
        meta.append(chunks[i].get_meta(audioSet,options['task']))

    return data, meta

def asynchronous_learning(audioSet, 
                          audioOptions, 
                          nb_frames, 
                          model_options, 
                          model_name,
                          task = "genre", 
                          freq_bins = 168, 
                          batch_size = 10, 
                          nb_epochs = 50,
                          transfer_learning = False,
                          model_base_name = None):
    '''
    Trains a dielemann model using asynchronous processing for loading tracks. 
    The training rely on a audioSet as defined in DatasetAudio.
    
    Parameters
    ----------
    audioSet:
        DatasetAudio: dataset to train our model
    audioOptions: 
        dict: our dataset options and transform options
    nb_frames:
        int: target resampling size for each chunk
    model_options:
        dict: our model options: see similarity_learning/models/dielemann/build 
              for infos
    model_name:
        string: name of our model
    task (optionnal):
        string: type of metadata used to train the model
    freq_bins (optionnal):
        do not change
    batch_size(optionnal):
        int: number of tracks per batch
    nb_epochs(optionnal):
        int: number of epochs for training. The algorithm uses early stopping 
             for optimisation, changing this value is not necessary unless required 
    '''

    print('batch_size:'+str(batch_size))
    asyncTask = AsynchronousTask(chunk_loader, 
                                 numWorkers = 2, 
                                 batchSize = batch_size, 
                                 shuffle = True)
    #add options
    options = audioOptions
    options["audioSet"] = audioSet
    options["task"] = task
    options["frames number"] = nb_frames
    alphabet_size = len(set(audioSet.metadata[task]))
    model_options["Alphabet size"] = alphabet_size
    
    #build dielemann model
    if transfer_learning == False:
        model_base = build.build_conv_layers(nb_frames, freq_bins, model_options)
        model_full = build.add_fc_layers(model_base, model_options)
    elif transfer_learning == True and model_base_name != None:
        # pdb.set_trace()
        model_base_path = './similarity_learning/models/dielemann/saved_models/'+model_base_name+'_base.h5'
        model_base = keras.models.load_model(model_base_path)
        model_name = model_base_name + model_name
        model_full = build.add_fc_layers(model_base, model_options)

    history_list = {}
    
    #start training asynchronous
    for epoch in range(nb_epochs):
        asyncTask.createTask(audioSet.files, options)
        print('Epoch #' + str(epoch));
        for batchIDx, (currentData, currentMeta) in enumerate(asyncTask):
            #prevent errors from tracks with no downbeat annotations
            a = len(currentData)
            if a !=0:

                print('[Batch ' + str(batchIDx) + '] Learning step on ' + str(len(currentData[0])) + ' examples');
                x_train, y_train = reshape_data(currentData, currentMeta, alphabet_size);
                history = model_full.fit(x_train, y_train, batch_size = batch_size, epochs = 1, verbose = 1, validation_split = 0.2)
                history_list['epoch '+str(epoch)+' batch '+str(batchIDx)] = history.history
                x_train = 0
                y_train = 0
                currentData = 0
                currentMeta = 0
        '''
        '''
        print('Finished epoch #'+str(epoch))
              
        #early stopping based on validation loss, change patience to increase 
        #or decrease waiting time until stopping
        if epoch == 0:
            model_full_saved = model_full
            model_base_saved = model_base
            history_list_saved = history_list
            epoch_saved = 0
            patience = 10
            wait_time = 0
        else:
            val_loss_saved = 0
            val_loss = 0

            for i in range(int(np.floor(len(audioSet.files)/batch_size))):
                if "epoch "+str(epoch_saved)+" batch "+str(i) in history_list_saved and "epoch "+str(epoch)+" batch "+str(i) in history_list:
                    val_loss_saved = val_loss_saved + history_list_saved["epoch "+str(epoch_saved)+" batch "+str(i)]['val_loss'][0]
                    val_loss = val_loss + history_list["epoch "+str(epoch)+" batch "+str(i)]['val_loss'][0]
            if val_loss_saved < val_loss:
                wait_time = wait_time + 1
                print("wait time: "+str(wait_time))
            else:
                wait_time = 0
                model_full_saved = model_full
                model_base_saved = model_base
                history_list_saved = history_list
                epoch_saved = epoch

        if wait_time == patience:
            break

    print('train ok')
    save_model(model_full_saved, model_base_saved, model_options, history_list_saved, model_name)

def reshape_data(currentData, currentMeta, alphabet_size):
    '''
    Reshape data coming from the chunk_loader function to fit the model's 
    expectations
    
    Parameters
    ----------
    currentData:
        array: data to process
    currentMeta:
        list: metadata associated (expected output of the model)
    alphabet_size:
        int: number of different outputs possible
    '''
    batch_size = currentData[0].shape[0]
    data = np.zeros(((len(currentData))*batch_size, currentData[0].shape[1], currentData[0].shape[2]))
    meta = np.zeros((len(currentMeta))*batch_size)
    for i in range(len(currentData)):
        data[i*batch_size:i*batch_size +batch_size] = currentData[i]
        meta[i*batch_size:i*batch_size + batch_size] = currentMeta[i]

    x_train = np.swapaxes(np.array(data),1,2)
    y_train = np.array(meta)
    y_train = keras.utils.to_categorical(y_train, alphabet_size)
    return x_train, y_train

def save_model(model_full,
              model_base,
              model_options,
              history,
              name,
              pathmodel='./similarity_learning/models/dielemann/saved_models/'):
    '''
    Save a model, its base, its otpions and its history in the required
    directory. For each new model name, a new repositiry will be created

    Parameters
    ----------
    model_full: keras.model
        Model to save
    model_base: keras.model
        Base CNN of the model
    pathmodel (optionnal): string
        Path to the models repository
    '''
    
    print('Save model to ...')
    filepath_base = pathmodel + name + '_base.h5'
    filepath_full = pathmodel + name + '_full.h5'
    filename_history = pathmodel + name + '_history.pkl'
    filename_options = pathmodel + name + '_options'

    try:
        f = open(filepath_base, 'rb')
        f.close()
        resp = input('Model already exist, do you wish to overwrite it? y/n \n')
        if resp == 'y':
            print('Save base model as ' + name + '_base.h5' + '...')
            model_base.save(filepath_base)
            print('Base model saved in '+ filepath_base)

            print('Save full model as ' + name + '_full.h5' + '...')
            model_full.save(filepath_full)
            print('Base model saved in '+ filepath_full)

            print('Save history as ' + name + '_history...')
            file_history = open(filename_history, 'wb')
            pickle.dump(history, file_history)
            print('History saved in' + filename_history)

            print('Save options as ' + name + '_options ...')
            file_options = open(filename_options, 'wb')
            pickle.dump(model_options, file_options)

            file_history.close()
            file_options.close()


        else:
            resp = input('Change model name? y/n \n')
            if resp == 'y':
                name = input('Model name: ')
                filepath_base = pathmodel + name + '_base.h5'
                filepath_full = pathmodel + name + '_full.h5'
                filename_history = pathmodel + name + '_history' + '.pkl'

                print('Save base model as ' + name + '_base.h5' + '...')
                model_base.save(filepath_base)
                print('Base model saved in '+ filepath_base)

                print('Save full model as ' + name + '_full.h5' + '...')
                model_full.save(filepath_full)
                print('Base model saved in '+ filepath_full)

                print('Save history as ' + name + '_history...')
                file_history = open(filename_history, 'wb')
                pickle.dump(history, file_history)
                print('History saved in' + filename_history)

                print('Save options as ' + name + '_options ...')
                file_options = open(filename_options, 'wb')
                pickle.dump(model_options, file_options)

                file_history.close()
                file_options.close()
            else:
                print('Model not saved')
    except IOError:
        print('Save base model as ' + name + '_base.h5' + '...')
        model_base.save(filepath_base)
        print('Base model saved in '+ filepath_base)

        print('base model saved ok')
        print('Save full model as ' + name + '_full.h5' + '...')
        model_full.save(filepath_full)
        print('Base model saved in '+ filepath_full)

        print('Save history as ' + name + '_history.pkl ...')
        file_history = open(filename_history, 'wb')
        pickle.dump(history, file_history)
        print('History saved in' + filename_history)

        print('Save options as ' + name + '_options ...')
        file_options = open(filename_options, 'wb')
        pickle.dump(model_options, file_options)

        file_history.close()
        file_options.close()
        pass
