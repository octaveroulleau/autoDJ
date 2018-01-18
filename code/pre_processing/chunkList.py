#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Class chunks list

@author: pierre-amaury
"""
import chunkAudio

class ChunkList:
    def __init__(self):
        self.chunks_list = []        
    
    def get_all_track_id(self):
        
    def get_all_genres(self):
        genres = []
        for i in range(len(self.chunks_list)-1):
            genre = chunks_list[i].get_genre()
            genres.append(genre)
    
    def get_all_tempo(self):
        tempos = []
        for i in range(len(self.chunks_list)-1):
            tempo = chunks_list[i].get_tempo()
            tempo.append(tempo)
        
    def get_all_raw_audio_path(self):
        paths = []
        for i in range(len(self.chunks_list)-1):
            path = chunks_list[i].get_raw_audio_path()
            paths.append(path)