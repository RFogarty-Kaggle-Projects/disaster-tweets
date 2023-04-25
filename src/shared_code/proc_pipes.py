

import re

import pandas as pd
import keras
import sklearn.base


#Base/template class
class TextPipeline(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
	""" Template class for pipelines dealing with various text-processing

	"""

	#Should rarely require a fit step
	def fit(self, inpX, y=None):
		return self

	def transform(self, inpX):
		raise NotImplementedError("")

class _ReplaceTextInPlaceTemplate(TextPipeline):
	""" Template class for replacing text in place

	"""
	def __init__(self, inpCol="proc_text"):
		self.inpCol = inpCol

	def _getMapFunction(self):
		raise NotImplementedError("")

	def transform(self, inpX):
		inpX[self.inpCol] = inpX[self.inpCol].map( self._getMapFunction() )
		return inpX





class ReplaceHtmlSpace():
	
	def __init__(self, onFields):
		self.onFields = list(onFields)
	
	def fit(self, inpX, y=None):
		return self
	
	def transform(self, inpX):
		outX = inpX.copy()
		
		def _mapFunct(inpX):
			if pd.isna(inpX):
				return inpX
			else:
				return str(inpX).replace("%20"," ")
		
		for field in self.onFields:
			outX[field] = outX[field].map( _mapFunct)
		return outX
	

class AddLengthText():

	def __init__(self, outName="length_text", targCol="text"):
		self.outName = outName
		self.targCol = targCol

	def fit(self, inpX, y=None):
		return self

	def transform(self, inpX):
		outX = inpX.copy()
		outX[self.outName] = outX[self.targCol].map(_mapLenText)
		return outX

class AddNumbHyperlinks():

	def __init__(self, targCol="text", outCol="n_hyperlinks"):
		self.regex = re.compile("(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])")
		self.targCol = targCol
		self.outCol = outCol

	def fit(self, inpX, y=None):
		return self

	def transform(self, inpX):
		outX = inpX.copy()
		def _mapFunct(inpText):
			return len(self.regex.findall(inpText))
		outX[self.outCol] = outX[self.targCol].map(_mapFunct)
		return outX

class AddNumbWords():

	def __init__(self, outName="numb_words", targCol="text"):
		self.outName = outName
		self.targCol = targCol

	def fit(self, inpX, y=None):
		return self

	def transform(self, inpX):
		outX = inpX.copy()
		outX[self.outName] = outX[self.targCol].map( _mapNWords )
		return outX 

class AddMeanWordLength():

	def __init__(self, outName="word_length_mean", targCol="text"):
		self.outName = outName
		self.targCol = targCol

	def fit(self, inpX, y=None):
		return self

	def transform(self, inpX):
		outX = inpX.copy()
		nWords = outX[self.targCol].map(_mapNWords)
		textLength = outX[self.targCol].map(_mapLenText)
		outX[self.outName] = textLength / nWords
		return outX

def _mapNWords(inpText):
	return len(re.findall("\w+",inpText))

def _mapLenText(inpText):
	return len(inpText)


class AddNumbChars():
	
	def __init__(self, targChar, outName, targCol="text"):
		self.targChar = targChar
		self.targCol = targCol
		self.outName = outName
		
	def fit(self, inpX, y=None):
		return self
	
	def transform(self, inpX):
		outX = inpX.copy()
		def _mapFunct(inpText):
			return inpText.count(self.targChar)
		
		outX[self.outName] = outX[self.targCol].map(_mapFunct)
		return outX


class ReplaceHyperlinks():
	
	def __init__(self, targCol="text", replacement=" hyperlink "):
#		 self.regex = re.compile("/(?:(?:https?|ftp|file):\/\/|www\.|ftp\.)(?:\([-A-Z0-9+&@#\/%=~_|$?!:,.]*\)|[-A-Z0-9+&@#\/%=~_|$?!:,.])*(?:\([-A-Z0-9+&@#\/%=~_|$?!:,.]*\)|[A-Z0-9+&@#\/%=~_|$])/igm")
#		 self.regex = re.compile("http")
		self.regex = re.compile("(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])")
		self.targCol = targCol
		self.replacement = replacement
	
	def fit(self, inpX, y=None):
		return self
	
	def transform(self, inpX):
		outX = inpX.copy()
		def _mapFunct(inpStr):
			return self.regex.sub(self.replacement,inpStr)
		
		outX[self.targCol] = outX[self.targCol].map(_mapFunct)
		return outX


class ConvertToLowerCase():

	def __init__(self, targCol="text"):
		self.targCol = targCol

	def fit(self, inpX, y=None):
		return self

	def transform(self, inpX):
		outX = inpX.copy()
		outX[self.targCol] = outX[self.targCol].map(lambda x:x.lower())
		return outX

class RemoveHyperlinkPunctuation():
	
	def __init__(self, targCol="text"):
		self.regex = re.compile("(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])")
		self.targCol = targCol
	
	def fit(self, inpX, y=None):
		return self
	
	def transform(self, inpX):
		outX = inpX.copy()

		def _subHtml(matchObj):
			outStr = "".join( matchObj.groups() ).replace(".","").replace("/","")
			return outStr
		
		def _mapFunct(inpText):
			return re.sub(self.regex, _subHtml, inpText)
		
		outX["text"] = outX["text"].map(_mapFunct)
		return outX


class ReplaceRepeatedPunctuation():
	
	def __init__(self,repPunctList=None, targCol="text"):
		_defaultList = ["?",".","!"]
		self.repPunctList = _defaultList if repPunctList is None else repPunctList

	def fit(self, inpX, y=None):
		return self
	
	def transform(self, inpX):
		outX = inpX.copy()
		for punct in self.repPunctList:
			_regexPart = "[{}]".format(punct) + "{2,}"
			outX["text"] = outX["text"].map( lambda x: re.sub(_regexPart, punct, x)   )
		return outX


class RemoveLeadingPunctuation():
	
	def __init__(self, punctList=None, targCol="text"):
		self.punctList = self._getDefPunctList() if punctList is None else punctList
		self.targCol = targCol
		
	def _getDefPunctList(self):
		return [".",";",":","?","'", "!"]
	
	def fit(self, inpX, y=None):
		return self
	
	def transform(self, inpX):
		outX = inpX.copy()
		punctPart = "".join(self.punctList)
		pattern = "([" + punctPart + "])" + "([\w]+)"
		def _mapFunct(inpStr):
			return re.sub(pattern, "\\1 \\2",inpStr)
		
		outX[self.targCol] = outX[self.targCol].map(_mapFunct)
		return outX


class RemoveTrailingPunctuation():
	
	def __init__(self, punctList=None, targCol="text"):
		self.punctList = self._getDefPunctList() if punctList is None else punctList
		self.targCol = targCol
		
	def _getDefPunctList(self):
		return [".",";",":","?","'", "!"] + [","]
	
	def fit(self, inpX, y=None):
		return self
	
	def transform(self, inpX):
		outX = inpX.copy()
		punctPart = "".join(self.punctList)
		pattern = "([\w]+)([" + punctPart + "])"
		def _mapFunct(inpStr):
			return re.sub(pattern, "\\1 \\2",inpStr)
		
		outX[self.targCol] = outX[self.targCol].map(_mapFunct)
		return outX




class MapSingleDigitNumbersToWords():
	
	def __init__(self, targCol="text"):
		self.mapDict = self._loadDefDict()
		self.targCol = targCol
	
	def _loadDefDict(self):
		outDict = {"0":"zero", "1":"one", "2":"two", "3":"three", "4":"four",
				   "5":"five", "6":"six", "7":"seven", "8":"eight", "9":"nine"}
		return outDict
	
	def fit(self, inpX, y=None):
		return self
	
	def transform(self, inpX):
		outX = inpX.copy()
		pattern = "([ ;:?])([0-9])([ .?;:][^0-9])"
		
		def _mapRegex(inpMatch):
			groups = inpMatch.groups()
			midGroup = self.mapDict[ inpMatch.groups()[1] ]
			return groups[0] + midGroup + groups[-1]
		
		def _mapFunct(inpStr):
			return re.sub(pattern, _mapRegex, inpStr)
		
		outX[self.targCol] = outX[self.targCol].map(_mapFunct)
		return outX


class ReplaceSpecialHTMLStrings():
	
	def __init__(self, mapDict=None, targCol="text"):
		self.mapDict = self._getDefaultMapDict() if mapDict is None else mapDict
		self.targCol = targCol
	
	def _getDefaultMapDict(self):
		outDict = {"&amp;":"&", "&gt;":">", "&lt;":"<"}
		return outDict
	
	def fit(self, inpX, y=None):
		return self
	
	def transform(self, inpX):
		outX = inpX.copy()
		def _mapFunct(inpStr):
			outStr = inpStr
			for key in self.mapDict:
				outStr = outStr.replace(key, self.mapDict[key])
			return outStr
		
		outX[self.targCol] = outX[self.targCol].map(_mapFunct)
		return outX

class DropTextDuplicates():

	def __init__(self):
		pass

	def fit(self, inpX, y=None):
		return self

	def transform(self, inpX):
		outX = inpX.copy()
		return outX.drop_duplicates(["text"])

class AddVectorizedColumns():
	
	def __init__(self, vocabSize, seqWordLength, prefix="vect_text_"):
		self.vocabSize = vocabSize
		self.seqWordLength = seqWordLength
		self.prefix = prefix
		
	def fit(self, inpX, y=None):
		_currKwargs = {"max_tokens":self.vocabSize, "output_sequence_length":self.seqWordLength}
		self.vectorizer = keras.layers.TextVectorization(**_currKwargs)
		self.vectorizer.adapt(inpX["text"].to_numpy())
		return self
	
	def transform(self, inpX):
		_vectTextData = self.vectorizer(inpX["text"]).numpy()
		_vectTextCols = [self.prefix + "{}".format(x) for x in range(_vectTextData.shape[-1])]
		_extraFrame = pd.DataFrame(data=_vectTextData, columns=_vectTextCols, index=inpX.index)
		outX = pd.concat( [inpX.copy(),_extraFrame ],axis=1 )
		return outX

class ReplaceContractions():
	
	def __init__(self, contractionDict=None, targCol="text"):
		self.contractionDict = _decontractMaps if contractionDict is None else contractionDict
		
	def fit(self, inpX, y=None):
		return self
	
	def transform(self, inpX):
		outX = inpX.copy()
		def _mapFunct(inpX):
			outStr = inpX
			for key in self.contractionDict.keys():
				outStr = outStr.replace(key, self.contractionDict[key])
			return outStr
		
		outX["text"] = outX["text"].map(_mapFunct)
		return outX

_decontractMaps = {"i'm":"i am",
                   "it's":"it is",
                   "don't":"do not",
                   "can't":"can not",
                   "you're":"you are",
                   "that's":"that is",
                   "i've":"i have",
                   "i'll":"i will",
                   "he's":"he is",
                   "there's":"there is",
                   "didn't":"did not",
                   "i'd":"i did",
                   "what's":"what is",
                   "they're":"they are",
                   "isn't":"is not",
                   "we're":"we are",
                   "let's": "let us",
                   "won't": "will not",
                   "ain't":"is not",
                   "we're":"we are",
                   "reddit's":"reddit is",
                   "she's":"she is",
                   "wasn't":"was not",
                   "haven't":"have not",
                   "you'll":"you will",
                   "aren't":"are not",
                   "we've":"we have",
                   "wouldn't":"would not",
                   "you've":"you have",
                   "here's":"here is",
                   "it's":"it is",
                   "shouldn't":"should not",
                   "who's":"who is",
                   "we'll":"we will",
                   "would've":"would have",
                   "y'all":"you all",
                   "they've":"they have",
                   "you'd":"you would",
                   "doesn't":"does not"
                  }




class RemoveStandardStopWords(_ReplaceTextInPlaceTemplate):
	""" Removes various stop-words (words like "a", "the") from text. 

	"""
	def __init__(self, inpCol="text", useDict=None):
		""" Initializer
		
		Args:
			inpCol: (str) Column to operate on
			useDict: (dict) By default will load stop words I took from NLTK (179 words). If a custom dict, keys should be stop words, values should be anything EXCEPT None

		Notes:
			Can load the default dictionary with _getStandardEnglishStopWordDict()
				 
		"""                                         
		self.inpCol = inpCol
		self.useDict = useDict if useDict is not None else _getStandardEnglishStopWordDict()

	def _getMapFunction(self):
		stopDict = self.useDict
		def _mapFunct(inpText):
			return " ".join( [x for x in inpText.split(" ") if stopDict.get(x,None) is None] )
		return _mapFunct



def _getStandardEnglishStopWordDict():
	""" Taken from NLTK english stop words [nltk.corpus.stopwords.words('english')]"""
	stopList = ['i',
	            'me',
	            'my',
	            'myself',
	            'we',
	            'our',
	            'ours',
	            'ourselves',
	            'you',
	            "you're",
	            "you've",
	            "you'll",
	            "you'd",
	            'your',
	            'yours',
	            'yourself',
	            'yourselves',
	            'he',
	            'him',
	            'his',
	            'himself',
	            'she',
	            "she's",
	            'her',
	            'hers',
	            'herself',
	            'it',
	            "it's",
	            'its',
	            'itself',
	            'they',
	            'them',
	            'their',
	            'theirs',
	            'themselves',
	            'what',
	            'which',
	            'who',
	            'whom',
	            'this',
	            'that',
	            "that'll",
	            'these',
	            'those',
	            'am',
	            'is',
	            'are',
	            'was',
	            'were',
	            'be',
	            'been',
	            'being',
	            'have',
	            'has',
	            'had',
	            'having',
	            'do',
	            'does',
	            'did',
	            'doing',
	            'a',
	            'an',
	            'the',
	            'and',
	            'but',
	            'if',
	            'or',
	            'because',
	            'as',
	            'until',
	            'while',
	            'of',
	            'at',
	            'by',
	            'for',
	            'with',
	            'about',
	            'against',
	            'between',
	            'into',
	            'through',
	            'during',
	            'before',
	            'after',
	            'above',
	            'below',
	            'to',
	            'from',
	            'up',
	            'down',
	            'in',
	            'out',
	            'on',
	            'off',
	            'over',
	            'under',
	            'again',
	            'further',
	            'then',
	            'once',
	            'here',
	            'there',
	            'when',
	            'where',
	            'why',
	            'how',
	            'all',
	            'any',
	            'both',
	            'each',
	            'few',
	            'more',
	            'most',
	            'other',
	            'some',
	            'such',
	            'no',
	            'nor',
	            'not',
	            'only',
	            'own',
	            'same',
	            'so',
	            'than',
	            'too',
	            'very',
	            's',
	            't',
	            'can',
	            'will',
	            'just',
	            'don',
	            "don't",
	            'should',
	            "should've",
	            'now',
	            'd',
	            'll',
	            'm',
	            'o',
	            're',
	            've',
	            'y',
	            'ain',
	            'aren',
	            "aren't",
	            'couldn',
	            "couldn't",
	            'didn',
	            "didn't",
	            'doesn',
	            "doesn't",
	            'hadn',
	            "hadn't",
	            'hasn',
	            "hasn't",
	            'haven',
	            "haven't",
	            'isn',
	            "isn't",
	            'ma',
	            'mightn',
	            "mightn't",
	            'mustn',
	            "mustn't",
	            'needn',
	            "needn't",
	            'shan',
	            "shan't",
	            'shouldn',
	            "shouldn't",
	            'wasn',
	            "wasn't",
	            'weren',
	            "weren't",
	            'won',
	            "won't",
	            'wouldn',
	            "wouldn't"]
	stopDict = {key:0 for key in stopList}
	return stopDict

