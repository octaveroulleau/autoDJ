#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Chunk class definition

@author: pierre-amaury
"""
from data.sets.audio import importAudioData
import skimage.transform as skt

#%%

class ChunkAudio():
    
    def __init__(self, sampling_frequency, original_track_id , echantillon_debut , echantillon_fin):
        self.track_id = original_track_id
        self.ech_debut = echantillon_debut
        self.ech_fin = echantillon_fin    
        self.Fs = sampling_frequency
    

    def get_cqt(self, audioSet, audioOptions):
        """ Function that gives the cqt matrix of the chunk
        
        Parameters
        ----------
        audioSet : DatasetAudio instance
            the dataset from which the chunk comes
            
        Returns
        -------
        
        
        """
        original_cqt = audioSet.data[self.track_id]

        file_cqt = [audioSet.files[self.track_id]]
        original_cqt, trash = importAudioData(file_cqt, audioOptions)
        original_cqt = original_cqt[0]

        nbBins, nb_frames = original_cqt.shape[0], original_cqt.shape[1]
        track_len = 30 #in seconds
        
        frame_deb = self.ech_debut*nb_frames/float(self.Fs*track_len)
        frame_fin = self.ech_fin*nb_frames/float(self.Fs*track_len)


#        return frame_deb, frame_fin        
        chunk_cqt = original_cqt[:, int(frame_deb):int(frame_fin)]
        
        
        return chunk_cqt
    
        
    def get_raw_audio_path(self, audioSet):
        """ Function that gives the path to raw audio where this chunk comes from
        
        Parameters
        ----------
        audioSet : DatasetAudio
            dataset where the chunk comes from
            
        Returns
        -------
        string
            path of the raw audio
        """
        raw_audio_path = audioSet.files[self.track_id]
        return raw_audio_path
    
    def get_tempo(self, audioSet):
        """ Function that gives the tempo of the original track this chunk comes from
        
        Parameters
        ----------
        audioSet : DatasetAudio
            dataset where the chunk comes from
            
        Returns
        -------
        float
            tempo of the original track
        
        """
        chunk_tempo = audioSet.metadata["tempo"][self.track_id]
        return chunk_tempo
    
    def get_genre(self, audioSet):
        """ Function that gives the genre of the original track this chunk comes from
        
        Parameters
        ----------
        audioSet : DatasetAudio
            dataset where the chunk comes from
            
        Returns
        -------
        int
            id of the genre of the original track
        """
        chunk_genre = audioSet.metadata["genre"][self.track_id]
        return chunk_genre
    
    def get_meta(self, audioSet, meta):
        chunk_meta = audioSet.metadata[meta][self.track_id]
        return chunk_meta
    