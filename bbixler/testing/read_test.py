#!/usr/bin/python3
from spt3g import core

filename = '/data2/home/polarbear/bbixler/pb2b_chwp_acq/data/sag_e1.g3'

encoder = 0
irig = 0

def reader(frame):
	global encoder
	global irig
	if (frame.type == core.G3FrameType.Timepoint and 
			'chwp_encoder_clock' in frame.keys()):
		encoder += 1
	elif (frame.type == core.G3FrameType.Timepoint and 
			'chwp_irig_clock' in frame.keys()):
		#print(frame.keys())
		irig += 1
		try:
			print(frame['count'])
		except:		
			frame['count'] = irig
	#return [frame]

pipe = core.G3Pipeline()
pipe.Add(core.G3Reader, filename=filename)
pipe.Add(reader)
pipe.Add(reader)
pipe.Run(profile=True)

print('encoder frames', encoder)
print('irig frames', irig)
