import numpy as np
import pandas as pd
import math

def getConditionalProbability(word, className, df, noWordClassCondProb):
    try:
        return df[word][className]
    except:
        return noWordClassCondProb[className]
        
def getSentenceCondProb(wordsList, className, df, noWordClassCondProb, priorProbabilities):
    score = math.log10(priorProbabilities[className])
    for word in wordsList:
        score += math.log10(getConditionalProbability(word, className, df, noWordClassCondProb))
    return score

def getListOfSentenceCondProb(arrayOfTokenizedTitle, className, df, noWordClassCondProb, priorProbabilities):
    score = np.zeros(len(arrayOfTokenizedTitle))
    for index in range(len(score)):
        wordsList = arrayOfTokenizedTitle[index]
        score[index] = getSentenceCondProb(wordsList, className, df, noWordClassCondProb, priorProbabilities)
    return score


def generateCondClassProb(test_df, model, noWordClassCondProb):
    model_df = model[0]
    priorProbabilities = model[1]
    for className in noWordClassCondProb.keys():
        test_df[className] = getListOfSentenceCondProb(test_df['tokenized_title'], 
                                                       className, 
                                                       model_df, 
                                                       noWordClassCondProb,
                                                       priorProbabilities)
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