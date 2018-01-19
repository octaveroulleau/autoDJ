import librosa
from const import SR, TEMPO



class PieceOfTrack:

    def __init__(self, track_name, beginning, end, tempo):
        self.name = track_name
        self.t_in = beginning
        self.t_out = end
        self.tempo = tempo

    def __repr__(self):
        text = "Track " + str(self.name) + " from " + str(self.t_in) + \
            " to " + str(self.t_out) + " at " + str(self.tempo) + " bpm"
        return text

    def render(self, tempo_out = TEMPO):
        try:           
            y, sr = librosa.load(self.name)
            y = y[self.t_in:self.t_out]
            if sr != SR :
                raise ValueError("Sampling rates are not all equal to "+str(SR))
        except ValueError as error:
            print(error)

        if (tempo_out != 0) & (self.tempo != 0):
            factor = float(tempo_out)/float(self.tempo)
            print(factor,self)
            print(type(factor))
            y = librosa.effects.time_stretch(y, factor)

        return y.tolist()

    def fadein_render(self, tempo_out = TEMPO, bars = 1):
        t_fade = 60.0/self.tempo * 4*bars
        try:           
            y, sr = librosa.load(self.name)
            y = y[self.t_in-t_fade:self.t_out]
            if sr != SR :
                raise ValueError("Sampling rates are not all equal to "+str(SR))
        except ValueError as error:
            print(error)

        if (tempo_out != 0) & (self.tempo != 0):
            factor = float(tempo_out)/float(self.tempo)
            print(factor,self)
            print(type(factor))
            y = librosa.effects.time_stretch(y, factor)

        return y.tolist()
