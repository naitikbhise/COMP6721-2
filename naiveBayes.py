# -------------------------------------------------------
# Project #2 Hacker News Dataset Analysis
# Written by Naitik Bhise (40106507) and Paras Kapoor (40114178)
# For COMP 6721 Section FI – Fall 2019
# --------------------------------------------------------

from GenerateCorpusDataframe import *

# The function below returns conditional probability of a token given the vocabulary and class
def getConditionalProbability(word, className, wordClassCondProbDF):
    try:
        return wordClassCondProbDF[word][className]
    except:
        return 0
    
# The function below generates score of a sentence conditoned on class        
def getSentenceCondProb(wordsList, className, model):
    wordClassCondProbDF = model[0]
    priorProbabilities = model[1]
    score = priorProbabilities[className]
    for word in wordsList:
        score += getConditionalProbability(word, className, wordClassCondProbDF)
    return score

# The function below list of scores given list of sentences conditoned on class  
def getListOfSentenceCondProb(arrayOfTokenizedTitle, className, model):
    score = np.zeros(len(arrayOfTokenizedTitle))
    for index in range(len(score)):
        wordsList = arrayOfTokenizedTitle[index]
        score[index] = getSentenceCondProb(wordsList, className, model)
    return score

# The function below generates scores of testing corpus conditoned on all the classes
def generateCondClassProb(test_df, model):
    priorProbabilities = model[1]
    for className in priorProbabilities.keys():
        if className in test_df.columns:
            test_df = test_df.drop(className, 1)
        test_df[className] = getListOfSentenceCondProb(test_df['tokenized_title'],className,model)
    return test_df

# The function below predicts the class of a sentence based on their scores
def generatePrediction(old_df,AllClasses):
    df = old_df[AllClasses].copy()
    return df.idxmax(axis=1)

# The function below creates a comparision column based on true class and predicted class
def comparePredictions(df,AllClasses):
    df['comparision'] = (df['predicted'] == df['Post Type'])
    cols = ['Title', 'predicted'] + AllClasses + ['Post Type', 'comparision']
    df = df[cols]
    return df

# The function below predicts class given a single sentence
def getRandomSentencePrediction(sentence,model):
    df = pd.DataFrame(data = {'Title' : [sentence]})
    df = addTokenizedColumnofTitle(df)
    df = generateCondClassProb(df, model)
    return generatePrediction(df,model[1].keys())[0]