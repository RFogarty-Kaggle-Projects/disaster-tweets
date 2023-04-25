


import numpy as np


def getGloveVectorsFromPath(inpPath):
	outDict = dict()
	with open(inpPath,"rt") as f:
		for line in f:
			splitLine = line.split()
			outDict[splitLine[0]] = np.array(splitLine[1:],dtype='float64')
	return outDict

