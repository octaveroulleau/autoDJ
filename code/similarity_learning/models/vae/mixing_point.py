class MixingPoint:

	def __init__(self, name1, time1, tempo1, name2, time2, tempo2):
		self.track1 = name1
		self.track2 = name2
		self.time1 = time1
		self.time2 = time2
		self.tempo1 = tempo1
		self.tempo2 = tempo2


	def __repr__(self):
		text = "Mix from track "+str(self.track1)+" at time "+str(self.time1)+" to track "+str(self.track2)+" at time "+str(self.time2)
		return text

