import librosa


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

    def render(self, tempo_out):
        y, sr = librosa.load(self.name)
        print(y.shape[0])
        print(self.t_out)

        try:
            y = y[self.t_in:self.t_out]
        except ValueError as error:
            print("PieceOfTrack : A chunk ends after the end of the track !")

        # y = y[self.t_in:self.t_out]

        if (tempo_out != 0) & (self.tempo != 0):
            factor = float(tempo_out)/float(self.tempo)
            print(factor,self)
            print(type(factor))
            y = librosa.effects.time_stretch(y, factor)

        return y.tolist(),sr
