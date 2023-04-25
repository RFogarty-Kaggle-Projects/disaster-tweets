
""" Convenience functions for generating standard combinations of pipelines (e.g. a standard pipeline for doing text-preprocessing) """

import sklearn as sk
import sklearn.pipeline

import proc_pipes as procPipeHelp
import train_pipes as trainPipeHelp

def loadTextPreprocPipeA(removeDuplicateTweets=True):
	_pipeComps = [ ("Convert text to lowercase", procPipeHelp.ConvertToLowerCase() ),
	               ("Expand contractions (e.g. \"we're\" to \"we are\"", procPipeHelp.ReplaceContractions() ),
	               ("Replace repeated punctuation (e.g. '???' to '?')", procPipeHelp.ReplaceRepeatedPunctuation() ),
	               ("Map single-digit numbers to words ('10' to 'ten')", procPipeHelp.MapSingleDigitNumbersToWords() ),
	               ("Remove hyperlinks", procPipeHelp.ReplaceHyperlinks(replacement=" ") ),
	               ("Replace some HTML characters", procPipeHelp.ReplaceSpecialHTMLStrings() ),
	               ("Remove trailing punctuation", procPipeHelp.RemoveTrailingPunctuation() ),
	               ("Remove leading punctuation", procPipeHelp.RemoveLeadingPunctuation() )
	] 

	if removeDuplicateTweets:
		_pipeComps.append(  ("Remove duplicate tweets", procPipeHelp.DropTextDuplicates()) )


	outPipe = sk.pipeline.Pipeline(_pipeComps)
	return outPipe


def loadGloveTransformsPipeA(embedDict):
	""" Adds min/max/mean embeddings for each dimension of the glove vectors (e.g. mean involves taking the average of each dimension of each word)
	
	Args:
		embedDict: (dict) Keys are words whilst values are embedding vectors
			 
	Returns
		outPipe: (sklearn Pipeline). Should require no fitting and works on pandas dataframe
 
	"""
	_pipeComps = [ ("Add min-GloVe embeddings" , trainPipeHelp.CreateMaxSentenceEncodings(embedDict) ),
	               ("Add max-GloVe embeddings" , trainPipeHelp.CreateMinSentenceEncodings(embedDict) ),
	               ("Add mean-GloVe embeddings", trainPipeHelp.CreateMeanSentenceEncodings(embedDict) ) ]

	outPipe = sk.pipeline.Pipeline(_pipeComps)
	return outPipe

