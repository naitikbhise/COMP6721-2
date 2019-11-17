import pandas as pd
import numpy as np
import math

def updateDictWords(dict_words, listOfTokens, postClass, AllClasses):
    for token in listOfTokens:
        if token not in dict_words:
            dict_words[token] = {}
            for className in AllClasses:
                dict_words[token][className] = 0
        dict_words[token][postClass] += 1
    return dict_words

def getWordFrequencyDataframe(df,AllClasses):
    dict_words = {}
    for index in range(len(df)):
        listOfTokens = df['tokenized_title'][index]
        postClass = df['Post Type'][index]
        dict_words = updateDictWords(dict_words, listOfTokens, postClass, AllClasses)
    return pd.DataFrame(dict_words)
        
def getTotalWordCount(wordsFrequencyDataframe):
    return np.sum(wordsFrequencyDataframe.sum(axis = 1, skipna = True)) 

def getLogOfProbability(probability):
    if probability == 0:
        logProbability = -1e5
    else:
        logProbability = math.log10(probability)
        logProbability = math.floor(logProbability*10**10)/10**10
    return logProbability

def convertProbToLog(probabilityVector):
    logVector = []
    for probability in probabilityVector:
        logVector.append(getLogOfProbability(probability))
    return logVector

def obtainDataframeWithClassProbabilities(old_df, AllClasses, delta, appendClassPrefix = 'prob_'):
    df = old_df.copy()
    df = df.transpose()
    uniqueWords = len(df)
    for className in AllClasses:
        wordsPerClass = np.sum(df[className])
        condClsLab = appendClassPrefix + className
        probabilityVector = df[className].map(lambda x: (int(x) + delta)/( wordsPerClass + delta*uniqueWords ))
        df[condClsLab] =  convertProbToLog(probabilityVector)
    df = df.transpose()
    return df

def getPriorProbabilities(df):
    priorProbabilities = {}
    classList = []
    for index in range(len(df)):
        if len(df['tokenized_title'][index]):
            classList.append(df['Post Type'][index])
    unique, counts = np.unique(classList, return_counts=True)
    for index in range(len(unique)):
        probability = counts[index]/np.sum(counts)
        priorProbabilities[unique[index]] = getLogOfProbability(probability)
    return priorProbabilities

def renameModelRows(df, AllClasses, appendClassPrefix):
    columnNamesExchange = {}
    for className in AllClasses:
        columnNamesExchange[appendClassPrefix + className] = className
    df = df.transpose()[columnNamesExchange.keys()]
    df = df.rename(columns=columnNamesExchange)
    df = df.transpose()
    return df

def getWordListBasedOnCount(words_df,maxCount = 1):
    wordList = []
    for word in list(words_df.columns.values):
        count = np.sum(words_df[word])
        if count <= maxCount:
            wordList.append(word)
    return wordList

def getWordListBasedOnPercent(words_df,Percent = 25):
    Counts = []
    Words = []
    for word in list(words_df.columns.values):
        Counts.append(np.sum(words_df[word]))
        Words.append(word)
    df = pd.DataFrame({"Count": Counts, "Words":Words})
    df = df.sort_values(by=['Count'], ascending=False)
    SortedWords = df['Words'].values.tolist()
    lastIndex = int(len(SortedWords)*(Percent/100))
    return SortedWords[0:lastIndex]