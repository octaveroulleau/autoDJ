""" Module assemble.
Mode normal (forward/compose/mix)
These function aim at assembling pieces of tracks that, according to the VAE, can be played one after another with smooth transition.
This module takes as input a list of mixing points (see mixing_point.py) proposed by the VAE.
It uses the audio files : .au and their signal metadata (downbeat, tonality) to produce an audio file that combine the chunks.
For this, it adjusts tonality and tempo using a phase vocoder, then aligns beats and downbeats of both extracts

@author: cyran

"""

import sys
sys.path.append("similarity_learning/models/vae/")
from piece_of_track import PieceOfTrack
import librosa
import numpy as np

def fetch_audio(mp_list):
    """ Step 1
    From the mixing points list, loads the audio and metadata relative to each interval between two mixing points.

    Parameters
    ----------
    mp_list : list of mixing points (see mixing_point.py)
        Defines the future mix.

    Returns
    -------
    tracklist : list of track (see vae/piece_of_track.py class) objects
        Audio objects.

    Example
    -------

    tracklist = fetch_audio(mp_list)

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

            if (beginning < 0):
                raise ValueError("A track starts before 0")

        except ValueError as error:
            print(error)

        cur_track = PieceOfTrack(mp_list[i].track1, beginning, end, cur_mp.tempo1)
        tracklist.append(cur_track)

    last_track = PieceOfTrack(mp_list[-1].track2,
                       mp_list[-1].time2, -1, mp_list[-1].tempo2)
    tracklist.append(last_track)

    return tracklist


def stack_tracks(tracklist):
    """ Step 2 : Concatenation of tracks from the tracklist with stretching
    
    Parameters
    ----------
    tracklist : list of track (see vae/piece_of_track.py class) objects
        Audio objects suitable to librosa.

    Returns
    -------
    final_set : librosa audio object
        Represents the final mix.
    sr : int
        The sample rate of the mix.

    Example
    -------

    final_set, sr = stack_tracks(tracklist)

    """

    final_set = []
    tempo = 120
    for piece in tracklist:
        current = piece.render(tempo)
        final_set = final_set + current[0] #data
        
    sr = current[1] #samplerate
    return final_set, sr


def mix_tracks(tracklist):
    """ Step 2bis : Sort of OLA to transition between tracks at each mixing point

    Parameters
    ----------
    tracklist : list of track (see vae/piece_of_track.py class) objects
        Audio objects suitable to librosa.

    Returns
    -------
    final_set : librosa audio object
        Represents the final mix.

    Example
    -------

    final_set = mix_tracks(tracklist)
    
    """

    # Constant gain.
    ##TODO
    final_set = []
    return final_set

def compose_track(mp_list):
    """ Assembles the new mix by calling the previous functions in the right order.
    
    Parameters
    ----------
    mp_list : list of mixing points (see mixing_point.py)
        Defines the future mix.
    
    Returns
    -------
    final_set : librosa audio object
        Represents the final mix.
    sr : int
        The sample rate of the mix.
    
    Example
    -------

    finalset, sr = compose_track(mp_list)

    """

    # Step 1 : fetch the audio 
    tracklist = fetch_audio(mp_list)
    # Step 2 : mix the tracks and merge
    concat_set, sr = stack_tracks(tracklist)
    #final_set = mix_tracks(concat_set)

    return np.array(concat_set), sr


def write_track(audio, sr, filename = 'finalmix.wav'):
    """ Writes created track to a file
    
    Parameters
    ----------
    audio : librosa audio object
        Represents the final mix.
    sr : int
        The sample rate of the mix.
    filename : str
        The path to write the audio file to (as '.wav')

    Returns
    -------
    None. Writes the output to disk.

    Example
    -------

    write_track(np.array(finalset),sr)

    """

    librosa.output.write_wav(filename, audio, sr)