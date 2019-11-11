import pandas as pd
import numpy as np


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


def obtainDataframeWithClassProbabilities(old_df, AllClasses, delta, appendClassPrefix = 'prob_'):
    df = old_df.copy()
    df = df.transpose()
    uniqueWords = len(df)
    absentWordCondProb = {}
    for className in AllClasses:
        wordsPerClass = np.sum(df[className])
        condClsLab = appendClassPrefix + className
        absentWordCondProb[condClsLab] = delta/( wordsPerClass + delta*uniqueWords )
        df[condClsLab] = df[className].map(lambda x: (int(x) + delta)/( wordsPerClass + delta*uniqueWords ))
    df = df.transpose()
    return df, absentWordCondProb