
""" Hacky code to wrap functions relevant for creation of neural nets """

import os

import numpy as np
import keras
import tensorflow as tf

import model_wrappers as modelWrapHelp

def createEmbedFixedLayer(vectorizer, embedDict, maxTextLength, retDetails=False):
	
	#Initializer
	_vocabSize = vectorizer.vocabulary_size()
	_embedDim = len(  next(iter(embedDict.values()))  )
	
	embedMatrix = np.zeros( (_vocabSize,_embedDim) )
	nHits, _missedWords = 0, list()

	#Create the embed matrix
	for idx,word in enumerate(vectorizer.get_vocabulary()):
		currVector = embedDict.get( word )
		if currVector is None:
			currVector = np.zeros(_embedDim)
			_missedWords.append(word)
		else:
			nHits += 1
		embedMatrix[idx,:] = currVector
	
	#Create a keras layer from the embed matrix
	_inputDim, _outputDim = _vocabSize, _embedDim
	_currKwargs = {"trainable":False, "embeddings_initializer":keras.initializers.Constant(embedMatrix),
				   "input_length":maxTextLength}
	outLayer = keras.layers.Embedding(_inputDim, _outputDim, **_currKwargs)
	
	#Create a "details" dictionary that can optionally be returned
	detailsDict = {"nHits":nHits, "nMisses":len(_missedWords), "missedWords":_missedWords}
	
	#
	if retDetails:
		return outerLayer, detailsDict
	else:
		return outLayer



#Not 1000% sure how the inherited functions will work but....
#should be fine
class BinaryNetWrapper( modelWrapHelp.GenericSklearnWrapper ):
	
	def __init__(self, feats, netCreator, saveFolder, targCol="target", featPrefix=None, valData=None,
				 valFract=0.2, optimizer="adam", loss="binary_crossentropy", metrics=None, nEpochs=10,
				 modelName="model_a", batchSize=None, verbose=0, logitsOutput=True):
		self.feats = feats
		self.netCreator = netCreator
		self.saveFolder = saveFolder
		self.nEpochs = nEpochs
		self.targCol = targCol
		self.featPrefix = featPrefix
		self.valData = valData
		self.valFract = valFract
		self.optimizer = optimizer
		self.loss = loss
		self.metrics = metrics
		self.modelName = modelName
		self.batchSize = batchSize
		self.verbose = verbose
		self.logitsOutput = logitsOutput
		
		#
		if self.logitsOutput:
			if self.loss == "binary_crossentropy":
				self.loss = keras.losses.BinaryCrossentropy(from_logits=True)

	def fit(self, inpX, y=None):
		#Get the data in correct format
		xFrame = self._getRelColsFromFrame(inpX)
		useX, useY = xFrame.to_numpy(), inpX[self.targCol].to_numpy()
#		 nInpts = useX.shape[1]
		
		#Create the neural net
		model = self._createModel()
		
		#Fit the net
		callbacks = self._createCallbacks()
		_currKwargs = {"epochs":self.nEpochs, "batch_size":self.batchSize, "callbacks":callbacks,
					   "validation_split":self.valFract, "verbose":self.verbose}
		if self.valData is not None:
			_currKwargs["validation_data"] = self._convValData( self.valData )		
		
		self.history = model.fit(useX, useY, **_currKwargs)

		#Set the model to the version with the BEST validation;
		#This is what we will use for predictions
		self.model = keras.models.load_model(self._getSavePath())
		
		return self
	
	def predict(self, inpX):
		probs = self.predict_proba(inpX)
		labels = np.where( probs >= 0.5, 1, 0) #TODO: Reduce dimension
		return labels.reshape(labels.shape[0])
		
	def predict_proba(self, inpX):
		self.model = keras.models.load_model(self._getSavePath())
		useX = self._getRelColsFromFrame(inpX).to_numpy()
		output = self.model.predict(useX)
		if self.logitsOutput:
			output = tf.sigmoid(output).numpy()
#		 else:
#			 output = output.numpy()
		return output
		
	def _createModel(self):
		outNet = self.netCreator.create()
		outNet.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
		return outNet

	def _createCallbacks(self):
		outCallbacks = list()
		
		#We save the best model to a file
		_currKwargs = {"monitor":"val_accuracy","save_best_only":True}
		_usePath = self._getSavePath()
		modelSaver = keras.callbacks.ModelCheckpoint(filepath=_usePath, save_best_only=True)
		outCallbacks.append(modelSaver)
		
		return outCallbacks
	
	def _convValData(self, valData):
		xFrame = self._getRelColsFromFrame(valData)
		useX, useY = xFrame.to_numpy(), valData[self.targCol].to_numpy()
		return useX, useY
		
	def _getSavePath(self):
		return os.path.join(self.saveFolder, self.modelName)




#Layers are:
#a) The embedding layer (which should be FIXED really)
#b) dropout layer
#c) Convolution Layer (a single one) 
#d) Pooling layer (single one) - max or average
#e) Global pooling layer (max or average)
#f) Dense hidden layers with dropout after the FIRST
#g) Dense output layer [single output since binary classification]
class SimpleConvNetFactoryA():
	
	def __init__(self, embedLayer, vectorLength, maxNumbVectors, dropoutFracts=None,
				 convNFilters=64, convFilterLength=5, convStride=1, convActFunct="elu",
				 poolSize=5, poolStride=1, poolTypeA="max", globPoolType="max", hiddenLayerNodes=None,
				 hiddenLayerActFunct="elu", outLayerActFunct="sigmoid"):
		self.embedLayer = embedLayer
		self.vectorLength = vectorLength
		self.maxNumbVectors = maxNumbVectors
		self.dropoutFract = [0.0,0.0] if dropoutFracts is None else dropoutFracts
		self.convNFilters = convNFilters
		self.convFilterLength = convFilterLength
		self.convStride = convStride
		self.convActFunct = convActFunct
		self.poolSize = poolSize
		self.poolTypeA = poolTypeA
		self.poolStride = poolStride
		self.globPoolType = globPoolType
		self.hiddenLayerNodes = list() if hiddenLayerNodes is None else hiddenLayerNodes
		self.hiddenLayerActFunct = hiddenLayerActFunct
		self.outLayerActFunct = outLayerActFunct
		
	def create(self):
		#Build a few of the key individual layers
		_firstDropout, _secondDropout = [keras.layers.Dropout(x) for x in self.dropoutFract]
		
		_currKwargs = {"filters":self.convNFilters, "kernel_size":self.convFilterLength,
					   "input_shape":(self.maxNumbVectors, self.vectorLength),
					   "strides":self.convStride,"activation":self.convActFunct}
		_convLayer = keras.layers.Conv1D(**_currKwargs)
		
		#Pooling
		_poolTypeFunctDict = {"max":keras.layers.MaxPooling1D, "average":keras.layers.AveragePooling1D}
		_poolLayerFunct = _poolTypeFunctDict[self.poolTypeA]
		_currKwargs = {"pool_size":self.poolSize, "strides":self.poolStride}
		_poolLayer = _poolLayerFunct(**_currKwargs)
		
		#Global Pooling
		_globPoolFunctDict = {"max":keras.layers.GlobalMaxPooling1D(),
							  "average":keras.layers.GlobalAveragePooling1D()}
		_globPoolLayer = _globPoolFunctDict[ self.globPoolType ]
		
		
		#Create the hidden dense layers
		hiddenLayers = list()
		if len(self.hiddenLayerNodes)==0:
			hiddenLayers.append(_secondDropout)
		
		for idx,nNodes in enumerate(self.hiddenLayerNodes):
			keras.layers.Dense(nNodes, activation=self.hiddenLayerActFunct)
			if idx==1:
				hiddenLayers.append(_secondDropout)
			
		
		#
		_outputLayer = keras.layers.Dense(1, activation=self.outLayerActFunct)
		
		#Put it all together
		allLayers = ( [self.embedLayer, _firstDropout, _convLayer, _poolLayer, _globPoolLayer] 
					  + hiddenLayers + [_outputLayer] )
		
		return keras.models.Sequential(allLayers)


