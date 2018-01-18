""" Module assemble.
Mode normal (forward)
These function aim at assembling pieces of track that, according to the VAE, can be played one after another with smooth transition.
This module takes as input a list of mixing points (see mixing_point.py) proposed by the VAE.
It uses the audio files : .au and their signal metadata (downbeat, tonality) to produce an audio file that combine the chunks.
For this, it adjusts tonality and tempo using a phase vocoder, then aligns beats and downbeats of both extracts
"""
from similarity_learning.models.vae.piece_of_track import PieceOfTrack
from re_synthesis.mixing_techniques import mix
import librosa
import numpy as np

def fetch_audio(mp_list):
    """ Step 1
    From the mixing points list, loads the audio and metadata relative to each interval between two mixing points.
    Returns a list of track (see track.py class) objects : tracklist
    """
    first_track = PieceOfTrack(mp_list[0].track1, 0, mp_list[
                        0].time1, mp_list[0].tempo1)
    tracklist = [first_track]

    for i in range(1, len(mp_list)):

        try:
            prev_mp = mp_list[i - 1]
            cur_mp = mp_list[i]

            if (cur_mp.track1 != prev_mp.track2):
                raise ValueError("A track lost its way")

            beginning = prev_mp.time2
            end = cur_mp.time1

            if (beginning > end):
                raise ValueError("A track ends before the beginning")

        except ValueError as error:
            print(error)

        cur_track = PieceOfTrack(mp_list[i].track1, beginning, end, cur_mp.tempo1)
        tracklist.append(cur_track)

    last_track = PieceOfTrack(mp_list[-1].track2,
                       mp_list[-1].time2, -1, mp_list[-1].tempo2)
    tracklist.append(last_track)

    return tracklist


def mix_tracks(tracklist):
    """ Step 2
    Concatenation of tracks from the tracklist with stretching
    """
    tempo = 120
    final_set = tracklist[0].render(tempo)[0]
    for piece in tracklist[1:]:
        current = piece.render(tempo)
        data = current[0]
        sr = current[1]
        final_set = mix(final_set,data,'basic')
        
        
    return final_set, sr




def compose_track(mp_list):
    """
    Input : a list of mixing points
    Step 1 : fetch the audio 
    Step 2 : mix the tracks and merge
    Returns the new audio track
    """
    tracklist = fetch_audio(mp_list)
    final_set, sr = mix_tracks(tracklist)
    #final_set = mix_tracks(tracklist)

    return np.array(final_set), sr


def write_track(audio, sr, filename = 'finalmix.wav'):
    """ Writes created track to a file
    """
    librosa.output.write_wav(filename, audio, sr)