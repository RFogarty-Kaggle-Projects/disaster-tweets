
import numpy as np
import pandas as pd

import sklearn as sk
import sklearn.base
import sklearn.decomposition
import sklearn.feature_extraction


class AddBagOfWords(sklearn.base.TransformerMixin):

	def __init__(self, vectKwargs=None, textField="text", colPrefix="bow_"):
		""" Initializer
		
		Args:
			vectKwargs: (dict) Any keywords to pass to "sk.feature_extraction.text.CountVectorizer"
			textField: (str) The field (in input dataframes) we apply the vectorization to
			colPrefix: (str)
				 
		"""
		vectKwargs = dict() if vectKwargs is None else vectKwargs
		self.vectorizer = sk.feature_extraction.text.CountVectorizer(**vectKwargs)
		self.textField = textField
		self.colPrefix = colPrefix

	def fit(self, inpX, y=None):
		self.vectorizer.fit(inpX[self.textField].tolist())
		return self

	def transform(self, inpX):
		outX = inpX.copy()
		countMatrix = self.vectorizer.transform(outX[self.textField].tolist())
		cols = [self.colPrefix + "{}".format(int(x)) for x in range(countMatrix.shape[1])]
		self._bowFrame = pd.DataFrame(countMatrix.todense(), columns=cols)
		self._bowFrame.index = outX.index
		return pd.concat([outX,self._bowFrame],axis=1)

class AddBagOfWords_TF_IDF():

	def __init__(self, vectKwargs=None, textField="text", colPrefix="tf_idf_"):
		vectKwargs = dict() if vectKwargs is None else vectKwargs
		self.vectorizer = sk.feature_extraction.text.TfidfVectorizer(**vectKwargs)
		self.textField = textField
		self.colPrefix = colPrefix

	def fit(self, inpX, y=None):
		self.vectorizer.fit(inpX[self.textField].tolist())
		return self

	def transform(self, inpX):
		outX = inpX.copy()
		countMatrix = self.vectorizer.transform(outX[self.textField].tolist())
		cols = [self.colPrefix + "{}".format(int(x)) for x in range(countMatrix.shape[1])]
		self._frame = pd.DataFrame(countMatrix.todense(), columns=cols)
		self._frame.index = outX.index
		return pd.concat([outX,self._frame],axis=1)


class _ReduceEmbeddingsTemplate():

	def fit(self, inpX, y=None):
		return self
	
	def transform(self, inpX):
		outX = inpX.copy()
		_nDims = len(self.embedDict["a"])
		_colNames = [self.prefix + "{}".format(x) for x in range(_nDims) ]
		newFrameData = np.zeros( (len(outX),_nDims)  )

		inpText = outX["text"].to_list()
		for rIdx,row in enumerate(inpText):
			newFrameData[rIdx] = self._getReducedVectorForTokens( row.split() )
		newFrame = pd.DataFrame(newFrameData, columns=_colNames, index=outX.index)
		return pd.concat([outX,newFrame],axis=1)

	def _getReducedVectorForTokens(self, tokens):
		raise NotImplementedError("")


class CreateMaxSentenceEncodings(_ReduceEmbeddingsTemplate):

	def __init__(self, embedDict, normalise=False, prefix="max_enc_"):
		self.embedDict = embedDict
		self.normalise = normalise
		self.prefix = prefix
		if normalise is True:
			raise NotImplementedError("")

	def _getReducedVectorForTokens(self, tokens):
		_nDims = len(self.embedDict["a"])
		outVect = np.zeros(_nDims)
		for token in tokens:
			try:
				currVect = np.array(self.embedDict[token])
			except KeyError:
				pass
			else:
				for idx,val in enumerate(outVect):
					if currVect[idx] > outVect[idx]:
						outVect[idx] = currVect[idx]
				
		return outVect


class CreateMinSentenceEncodings(_ReduceEmbeddingsTemplate):

	def __init__(self, embedDict, normalise=False, prefix="min_enc_"):
		self.embedDict = embedDict
		self.normalise = normalise
		self.prefix = prefix
		if normalise is True:
			raise NotImplementedError("")

	def _getReducedVectorForTokens(self, tokens):
		_nDims = len(self.embedDict["a"])
		outVect = np.zeros(_nDims)
		for token in tokens:
			try:
				currVect = np.array(self.embedDict[token])
			except KeyError:
				pass
			else:
				for idx,val in enumerate(outVect):
					if currVect[idx] < outVect[idx]:
						outVect[idx] = currVect[idx]
				
		return outVect



class CreateMeanSentenceEncodings(_ReduceEmbeddingsTemplate):
	""" Takes the mean of the vector-form for all words in the input text """
	
	def __init__(self, embedDict, normalise=False, prefix="mean_enc_"):
		self.embedDict = embedDict
		self.normalise = normalise
		self.prefix = prefix
		if normalise is True:
			raise NotImplementedError("")
	
#	def fit(self, inpX, y=None):
#		return self
#	
#	def transform(self, inpX):
#		outX = inpX.copy()
#		_nDims = len(self.embedDict["a"])
#		_colNames = [self.prefix + "{}".format(x) for x in range(_nDims) ]
#		newFrameData = np.zeros( (len(outX),_nDims)  )
#		for rIdx,row in enumerate(outX["text"]):
#			newFrameData[rIdx] = self._getMeanVectorForInpTokens( row.split() )
#		newFrame = pd.DataFrame(newFrameData, columns=_colNames)
#		return pd.concat([outX,newFrame],axis=1)
			
#	def _getMeanVectorForInpTokens(self, tokens):
	def _getReducedVectorForTokens(self, tokens):
		_nEmbed, _nDims = 0, len(self.embedDict["a"])
		outVect = np.zeros(_nDims)
		for token in tokens:
			try:
				currVect = np.array(self.embedDict[token])
			except KeyError:
				pass
			else:
				outVect += currVect
				_nEmbed += 1
		#Take the mean
		if _nEmbed > 0:
			outVect *= 1/_nEmbed
		return outVect


class FactorizeField():
	
	def __init__(self, fieldName, outName):
		self.fieldName = fieldName
		self.outName = outName
	
	def fit(self, inpX, y=None):
		keys = inpX[self.fieldName].unique()
		self.useDict = {key:val for val,key in enumerate(keys)}
		return self
	
	def transform(self, inpX):
		outX = inpX.copy()
		unknownVal = self.useDict[np.nan]
		outX[self.outName] = outX[self.fieldName].map( lambda x:  self.useDict.get(x,unknownVal)  )
		return outX

class MEncode():
	
	def __init__(self, useCol, mVal=1, targCol="target"):
		self.useCol = useCol
		self.targCol = targCol
		self.mVal = mVal
		
	def fit(self, inpX, y=None):
		self._globalMean = inpX[self.targCol].mean()
		self._groupInfo = inpX.groupby(self.useCol)[self.targCol].aggregate(["count","mean"])
		return self
	
	def transform(self, inpX):
		outFrame = inpX.copy()
		
		#
		_tempFrame = self._groupInfo.copy()
		outName = self.useCol + "_m{}".format(int(self.mVal))
		_tempFrame[outName] = 0
		
		def _applyFunct(inpRow):
			factor = inpRow["count"] / (inpRow["count"] + self.mVal)
			groupContrib = factor*inpRow["mean"]
			globContrib = (1-factor)*self._globalMean
			inpRow[outName] = groupContrib + globContrib
			return inpRow
		
		_tempFrame = _tempFrame.apply(_applyFunct, axis=1).reset_index()[[self.useCol,outName]]
		_currKwargs = {"left_on":self.useCol,"right_on":self.useCol,"how":"left"}
		outFrame = pd.merge(outFrame, _tempFrame, **_currKwargs)
		outFrame[outName] = outFrame[outName].map(lambda x: self._globalMean if pd.isna(x) else x  )
		
		return outFrame


class StandardScaler(sklearn.base.TransformerMixin):
	""" Basically a wrapper for the sklearn StandardScaler; this returns a data frame rather than a numpy array """
	
	def __init__(self, ignoreCols=None):
		self.ignoreCols = ignoreCols
	
	def fit(self, inpX, inpY=None):
		self.useDict = dict()
		for col in inpX.columns:
			inclVal = True
			if self.ignoreCols is not None:
				if str(col) in self.ignoreCols:
					inclVal = False
			if inclVal:
				self.useDict[col] = sk.preprocessing.StandardScaler()
				self.useDict[col].fit(inpX[col].to_numpy().reshape(-1, 1)  )

		return self
	
	def transform(self, inpX, inpY=None):
		outX = inpX.copy()
		for col in inpX.columns:
			inclVal = True
			if self.ignoreCols is not None:
				if str(col) in self.ignoreCols:
					inclVal = False
			if inclVal:
				outX[col] = self.useDict[col].transform( inpX[col].to_numpy().reshape(-1,1) )
		return outX



#
class AddPCA():
	
	def __init__(self, featsToUse, nComponents=None):
		self.nComponents = len(featsToUse) if nComponents is None else nComponents
		self.featsToUse = featsToUse
		
	def fit(self, inpX, y=None):
		self.pcaObj = sk.decomposition.PCA(n_components=self.nComponents)
		useFrame = inpX.copy()
		useFrame = useFrame[self.featsToUse]
		
		#Need to scale before doing PCA
		self.scaler = StandardScaler()
		self.scaler.fit(useFrame)
		useFrame = self.scaler.transform(useFrame)

		#
		self.pcaObj.fit(useFrame)
		
		return self
	
	def transform(self, inpX):
		#Setup frame for PCA
		useFrame = inpX.copy()
		useFrame = useFrame[self.featsToUse]
		useFrame = self.scaler.transform(useFrame)

		#Add components
		pcVals = self.pcaObj.transform(useFrame)
		colNames = ["pc_{}".format(x) for x in range(pcVals.shape[1])]
		pcFrame = pd.DataFrame(pcVals, index=useFrame.index, columns=colNames)
		
		#
		outFrame = inpX.copy()
		outFrame = outFrame.join( pcFrame )
		return outFrame

