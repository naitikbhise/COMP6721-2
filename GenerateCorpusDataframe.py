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
    if word in stopWordSet:
        return True
    else:
        return False
    
def tokenizeSentence(sentence):
    tokenized = []
    text = nltk.word_tokenize(sentence.lower())
    for word,pos in nltk.pos_tag(text):
        if checkPunctuation(word):
            continue    
        tag = penn_to_wn(pos)
        if tag is None:
            lemmatizedWord = word
        else:
            lemmatizedWord = wordnet_lemmatizer.lemmatize(word,tag)
        tokenized.append(lemmatizedWord)
    return tokenized

def addTokenizedColumnofTitle(data):
    if 'tokenized_title' in data.columns:
        data = data.drop('tokenized_title', 1)
    data['tokenized_title'] = data['Title'].map(lambda x:tokenizeSentence(x))
    return data

def filterByWordList(listOfTokens,stopWordSet):
    filteredList = []
    for token in listOfTokens:
        if checkStopWords(token,stopWordSet):
            continue            
        filteredList.append(token)
    return filteredList

def filterTokensByWordList(data,stopWordList):
    stopWordSet = set(stopWordList)
    if 'tokenized_title' in data.columns:
        data['tokenized_title'] = data['tokenized_title'].map(lambda x:filterByWordList(x,stopWordSet))
    else:
        return data
    return data

def filterByWordLength(listOfTokens):
    filteredList = []
    for token in listOfTokens:
        if len(token)<=2 or len(token)>=9:
            continue            
        filteredList.append(token)
    return filteredList

def filterTokensByWordLength(data):
    if 'tokenized_title' in data.columns:
        data['tokenized_title'] = data['tokenized_title'].map(lambda x:filterByWordLength(x))
    else:
        return data
    return data