from GenerateCorpusDataframe import *
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

def generatePrediction(old_df,AllClasses):
    df = old_df[AllClasses].copy()
    return df.idxmax(axis=1)

def comparePredictions(df,AllClasses):
    df['comparision'] = (df['predicted'] == df['Post Type'])
    cols = ['Title', 'predicted'] + AllClasses + ['Post Type', 'comparision']
    df = df[cols]
    return df

def getRandomSentencePrediction(sentence,model):
    df = pd.DataFrame(data = {'Title' : [sentence]})
    df = addTokenizedColumnofTitle(df)
    df = generateCondClassProb(df, model)
    return generatePrediction(df,model[1].keys())[0]