class PieceOfTrack:

    def __init__(self, track_id, beginning, end, tempo):
        self.id = track_id
        self.t_in = beginning
        self.t_out = end
        self.tempo = tempo

    def __repr__(self):
        text = "Track " + self.id + " from " + self.t_in + \
            " to " + self.t_out + " at " + self.tempo + " bpm"
        return text

    def render(self, tempo_out = self.tempo):
    	y = DATASET[self.id].data[self.t_in:self.t_out]
    	factor = tempo_out/self.tempo

    	stretched = librosa.effects.time_stretch(y, factor)

    	return stretched
