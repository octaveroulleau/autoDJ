""" Module assemble.
Mode normal (forward/compose/mix)
These function aim at assembling pieces of tracks that, according to the VAE, can be played one after another with smooth transition.
This module takes as input a list of mixing points (see mixing_point.py) proposed by the VAE.
It uses the audio files : .au and their signal metadata (downbeat, tonality) to produce an audio file that combine the chunks.
For this, it adjusts tonality and tempo using a phase vocoder, then aligns beats and downbeats of both extracts
"""

import sys
sys.path.append("similarity_learning/models/vae/")
sys.path.append("re_synthesis/")
from piece_of_track import PieceOfTrack
from mixing_techniques import create_mix

import librosa
import numpy as np
from operator import add,mul
from const import SR, TEMPO


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

        except ValueError as error:
            print(error)

        cur_track = PieceOfTrack(mp_list[i].track1, beginning, end, cur_mp.tempo1)
        tracklist.append(cur_track)

    last_track = PieceOfTrack(mp_list[-1].track2,
                       mp_list[-1].time2, -1, mp_list[-1].tempo2)
    tracklist.append(last_track)

    return tracklist



def create_mix(tracklist, style = 'basic'):
    """ Step 2 : Create mix of tracks from the tracklist with stretching
    
    Parameters
    ----------
    tracklist : list of track (see vae/piece_of_track.py class) objects
        Audio objects suitable to librosa.
    style : the style of the mix (basic, noise or Xfade)

    Returns
    -------
    final_set : librosa audio object
        Represents the final mix.

    Example
    -------

    dj_set = stack_tracks(tracklist, 'noise')

    """
    
    dj_set = tracklist[0].render()

    for piece in tracklist[1:]:
        if (style == 'basic'):
            dj_set = dj_set + piece.render()
            
        elif (style == 'noise'):
            length = int(1.2*SR)
            
            ramp = [float(i)/(length*3) for i in range(length)]
            tail = map(mul,np.random.rand(length),ramp)
            dj_set[-length:] = map(add,dj_set[-length:],tail)
            dj_set = dj_set + piece.render()
        elif (style == 'Xfade'):
            fade_bars = 1 #bars
            fade_len = 4*fade_bars * 60.0/TEMPO
            to_add = piece.fadein_render(fade_bars)
            
            dj_set[-fade_len:] = map(add,dj_set[-fade_len:],to_add)
            dj_set = dj_set + piece.render()
            
            
    return dj_set


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
    
    Example
    -------

    finalset = compose_track(mp_list)

    """

    # Step 1 : fetch the audio 
    tracklist = fetch_audio(mp_list)
    final_set= create_mix(tracklist)

    return np.array(final_set)



def write_track(audio, filename = 'finalmix.wav'):
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

    write_track(np.array(finalset))

    """

    librosa.output.write_wav(filename, audio, SR)
