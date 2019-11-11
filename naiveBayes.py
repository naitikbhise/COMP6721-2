import numpy as np
import pandas as pd
import math

def getConditionalProbability(word, className, wordClassCondProbDF):
    try:
        return wordClassCondProbDF[word][className]
    except:
        return 1
        
def getSentenceCondProb(wordsList, className, model):
    wordClassCondProbDF = model[0]
    priorProbabilities = model[1]
    score = math.log10(priorProbabilities[className])
    for word in wordsList:
        score += math.log10(getConditionalProbability(word, className, wordClassCondProbDF))
    return score

def getListOfSentenceCondProb(arrayOfTokenizedTitle, className, model):
    score = np.zeros(len(arrayOfTokenizedTitle))
    for index in range(len(score)):
        wordsList = arrayOfTokenizedTitle[index]
        score[index] = getSentenceCondProb(wordsList, className, model)
    return score


def generateCondClassProb(test_df, model):
    priorProbabilities = model[1]
    for className in priorProbabilities.keys():
        test_df[className] = getListOfSentenceCondProb(test_df['tokenized_title'],className,model)
    return test_df

def comparePredictions(old_df,AllClasses):
    columnNamesExchange = {}
    for className in AllClasses:
        columnNamesExchange['prob_' + className] = className
    df = old_df[columnNamesExchange.keys()].copy()
    df = df.rename(columns=columnNamesExchange)
    df['predicted'] = df.idxmax(axis=1)
    df = pd.concat([df, old_df[['Title', 'Post Type']]], axis=1)
    df['comparision'] = (df['predicted'] == old_df['Post Type'])
    cols = ['Title', 'predicted'] + AllClasses + ['Post Type', 'comparision']
    df = df[cols]
    return df