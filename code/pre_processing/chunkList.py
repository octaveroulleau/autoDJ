#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Class chunks list

@author: pierre-amaury
"""
import chunkAudio as ca

class ChunkList:
    def __init__(self):
        self.list_of_chunks = []  
        
    def add_chunk(self, chunk):
        """ Function that add a ChunkAudio instance in the attribute list_of_chunks
        
        Parameters
        ----------
        chunk : ChunkAudio instance
            chunk audio of a track to add to the list
            
        """
        self.list_of_chunks.append(chunk)
        
    def get_length(self):
        """ Function that returns the length of the list of chunks
        
        Returns
        -------
        int
            Amount of chunks in list_of_chunks
        
        """
        list_length = len(self.list_of_chunks)
        return list_length
    
    # get metadata
    def get_all_track_id(self):
        """ Function that gives all the track_id from where come from the chunks of list_of_chunks (in the same order)
        
        Returns
        -------
        int list
            track_id of each chunk in the same order as list_of_chunks
        """
        
        track_ids = []
        for i in range(len(self.list_of_chunks)-1):
            track_id = self.list_of_chunks[i].track_id
            track_ids.append(track_id)
        return track_ids
        
    def get_all_genres(self, audioSet):
        """ Function that gives the genre of all the chunks of list_of_chunks (in the same order)
        
        Returns
        -------
        int list
            genre of the track from which each chunk comes from, in the same order as list_of_chunks
        """
        
        genres = []
        for i in range(len(self.list_of_chunks)-1):
            genre = self.list_of_chunks[i].get_genre(audioSet)
            genres.append(genre)
        return genres
    
    def get_all_tempo(self, audioSet):
        """ Function that gives the tempo of all the chunks of list_of_chunks (in the same order)
        
        Returns
        -------
        int list
            tempo of the track from which each chunk comes from, in the same order as list_of_chunks
        """
        
        tempos = []
        for i in range(len(self.list_of_chunks)-1):
            tempo = self.list_of_chunks[i].get_tempo(audioSet)
            tempos.append(tempo)
        return tempos
        
    def get_all_raw_audio_path(self, audioSet):
        """ Function that gives the paths of the raw audio from where come from the chunks of list_of_chunks (in the same order)
        
        Returns
        -------
        string list
            path of the raw audio of the track from which each chunk comes from, in the same order as list_of_chunks
        """
        
        paths = []
        for i in range(len(self.list_of_chunks)-1):
            path = self.list_of_chunks[i].get_raw_audio_path(audioSet)
            paths.append(path)
        return paths
    