import pandas as pd
import numpy as np

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('wordnet')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except:
    nltk.download('averaged_perceptron_tagger')

from nltk.corpus import wordnet as wn
wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()


def dater(datetim,integer):
    date,time = datetim.split(" ")
    year,month,day = date.split("-")
    hour,minute,second = time.split(":")
    second = int(second)+ int(minute)*60 + int(hour)*60*60
    if integer==1:
        return year
    elif integer==2:
        return month
    elif integer==3:
        return day
    else:
        return second

def getDataframe(year):
    data = pd.read_csv("hn2018_2019.csv")
    data["year"] = data['Created At'].map(lambda x: dater(x,1))
    data["month"] = data['Created At'].map(lambda x: dater(x,2))
    data["day"] = data['Created At'].map(lambda x: dater(x,3))
    data["second"] = data['Created At'].map(lambda x: dater(x,4))
    return data[data["year"]==str(year)][['Title','Post Type','Number of Comments','Points','Author']].copy()

def penn_to_wn(tag):
    nltk_wn_pos = {'J':wn.ADJ,'V':wn.VERB,'N':wn.NOUN,'R':wn.ADV}
    try:
        return nltk_wn_pos[tag[0]]
    except:
        return None

def checkPunctuation(word):
    punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/','”','“','–',"'s",
                ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    if word in punctuations:
        return True
    else:
        return False

def checkStopWords(word,stopWordSet):
    if stopWordSet is None:
        return False
    if word in stopWordSet:
        return True
    else:
        return False
    
def tokenizeSentence(sentence,stopWordSet=None,filterByLength=False):
    tokenized = []
    text = nltk.word_tokenize(sentence.lower())
    for word,pos in nltk.pos_tag(text):
        if checkPunctuation(word):
            continue    
        if filterByLength:
            if len(word)<=2 or len(word)>=9:
                continue
        tag = penn_to_wn(pos)
        if tag is None:
            lemmatizedWord = word
        else:
            lemmatizedWord = wordnet_lemmatizer.lemmatize(word,tag)
        if checkStopWords(lemmatizedWord,stopWordSet):
            continue            
        tokenized.append(lemmatizedWord)
    return tokenized

def addTokenizedColumnofTitle(data,stopWordList=None,filterByLength=False):
    stopWordSet = None
    if stopWordList is not None:
        stopWordSet = set(stopWordList)
    if 'tokenized_title' in data.columns:
        data = data.drop('tokenized_title', 1)
    data['tokenized_title'] = data['Title'].map(lambda x:tokenizeSentence(x,stopWordSet,filterByLength))
    return data

def getPriorProbabilities(df):
    priorProbabilities = {}
    classList = []
    for index in range(len(df)):
        if len(df['tokenized_title'][index]):
            classList.append(df['Post Type'][index])
    unique, counts = np.unique(classList, return_counts=True)
    for index in range(len(unique)):
        priorProbabilities[unique[index]] = counts[index]/np.sum(counts)
    return priorProbabilities