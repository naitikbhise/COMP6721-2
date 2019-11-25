# -------------------------------------------------------
# Project #2 Hacker News Dataset Analysis
# Written by Naitik Bhise (40106507) and Paras Kapoor (40114178)
# For COMP 6721 Section FI – Fall 2019
# --------------------------------------------------------

import pandas as pd
import numpy as np

from posTagger import *

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

def getDataframe(year,filename):
    data = pd.read_csv(filename)
    data["year"] = data['Created At'].map(lambda x: dater(x,1))
    data["month"] = data['Created At'].map(lambda x: dater(x,2))
    data["day"] = data['Created At'].map(lambda x: dater(x,3))
    data["second"] = data['Created At'].map(lambda x: dater(x,4))
    return data[data["year"]==str(year)][['Title','Post Type','Number of Comments','Points','Author']].copy()


def checkStopWords(word,stopWordSet):
    if word in stopWordSet:
        return True
    else:
        return False
    
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