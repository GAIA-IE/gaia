
import pickle
import sys
#origin_name = pickle.load(open('results/name.p', 'rb'))
#print reader
inputdir = sys.argv[1]
outputName = sys.argv[2]
from glob import glob

i=0
bbox = {}
for name in glob('datasets/m18_a/bounding_boxes_*'):
	file1 = open(name)
	
	for line in file1:
		i+=1
		#if i == 10:
		#	break
		#print line
		data = line.split()
		#print data
		if len(data)>1:
			#print data
			if data[-1][-1] == 'g':
				continue		

			bbox[data[0]] = [int(data[-4]),int(data[-3]),int(data[-2]),int(data[-1])]
with open('results/'+outputName+'.pickle', 'wb') as handle:
    pickle.dump(bbox, handle, protocol=pickle.HIGHEST_PROTOCOL)
#print(bbox)