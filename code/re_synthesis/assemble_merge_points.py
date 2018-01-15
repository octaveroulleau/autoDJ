""" Module assemble.
These function aim at assembling audio chunks that, according to the VAE, can be played one after another with smooth transition.
This module takes as input a list of chunks IDs proposed by the VAE.
It uses the chunks (actual audio files : .au) and their signal metadata (downbeat, tonality) to produce an audio file that compile the chunks of the list.
For this, it adjusts tonality and tempo using a phase vocoder.
"""

def find mixing_points(mp_list):
	"""
	From the chunks list and their metadata, defines at which indexes there is a mixing point
	(ie the two chunks originate from different tracks)
	Returns a list of indexes such that for each i in the list, chunk i and chunk i+1 come from different tracks
	"""

def merge_small_chunks(audio_chunks, metadatas):
	""" For chunks that come from the same track initially : 
	Simple OLA to merge their audio into one big chunk.
	"""

def merge_large_chunks(audio_chunks, metadatas):
	""" For chunks that come from different tracks than their neighbors in the list :
	Implementaion of a phase vocoder to homogenize tempo and tonality
	Iterate on all pairs audio_chunks[i], audio_chunks[i+1]
	"""

def compose_track(mp_list):
	"""
	Input : a list of chunks IDs
	First finds the mixing points : mix_points_idxs
	First pass : merges chunks between mixing point (coming from the same track) with a simple ola + concatenate metadata info
	Second pass : merge large chunks with each other using the vocoder
	Returns the new audio track
	"""


def align_2_chunks(audio_chunks, metadatas):
	""" Aligns the 2 chunks temporally using their beat information
	"""


def mix_2_chunks(audio_chunks, metadatas):
	""" Sort of OLA to transition between two merged and aligned chunks
	"""
