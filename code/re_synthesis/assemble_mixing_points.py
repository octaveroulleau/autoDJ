""" Module assemble.
Mode normal (forward)
These function aim at assembling audio chunks that, according to the VAE, can be played one after another with smooth transition.
This module takes as input a list mixing points proposed by the VAE.
The mixing points are an ordonned sequence of quadruplets (trackID1, trackID2, time1, time2).
It uses the audio files : .au and their signal metadata (downbeat, tonality) to produce an audio file that combine the chunks.
For this, it adjusts tonality and tempo using a phase vocoder, then aligns beats and downbeats of both extracts
"""

def fetch_audio_meta(mp_list):
	""" Step 1
	From the mixing points list, loads the audio and metadata relative to each interval between two mixing points.
	Returns a list of (audio, beats) chunks : chunks_list
	"""

def merge_two_chunks(chunks_list):
	""" Step 2
	For chunks that come from different tracks than their neighbors in the list :
	Implementaion of a phase vocoder to homogenize tempo (and tonality)
	Iterate on all pairs chunks_list[i], chunks_list[i+1]
	"""


def align_chunks(chunks_list):
	""" Step 3
	Aligns the chunks of the list temporally using their beat information
	"""


def mix_two_chunks(audio_chunks, metadatas):
	""" Step 4
	Sort of OLA to transition between two homogene and aligned chunks
	Constant gain.
	"""

def compose_track(mp_list):
	"""
	Input : a list of mixing points
	Step 1 : fetch the audio and metadata related to each chunk
	Step 2 : for each chunk, change the tempo to homogenize the whole track (vocodeur)
	Step 3 : Align chunks with each other (beats + downbeats)
	Step 4 : Merge all chunks
	Returns the new audio track
	"""

def write_track(audio, filename):
	""" Writes created track to a file
	"""
