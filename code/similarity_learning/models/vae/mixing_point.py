class MixingPoint:

	def __init__(self, id1, time1, tempo1, id2, time2, tempo2):
    	self.track1 = id1
    	self.track2 = id2
    	self.time1 = time1
    	self.time2 = time2
    	self.tempo1 = tempo1
    	self.tempo2 = tempo2


    def __repr__(self):
    	text = "Mix from track "+self.track1+" at time "+self.time1+" to track "+self.track2+" at time "+self.time2
    	return text

