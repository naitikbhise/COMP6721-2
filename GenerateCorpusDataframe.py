# -------------------------------------------------------
# Project #2 Hacker News Dataset Analysis
# Written by Naitik Bhise (40106507) and Paras Kapoor (40114178)
# For COMP 6721 Section FI – Fall 2019
# --------------------------------------------------------

import pandas as pd
import numpy as np

from posTagger import *

# The function below retrun year,day,month,second separately from a string
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

# The function reads titles table from csv and returns a subset based on the given year
def getDataframe(year,filename):
    data = pd.read_csv(filename)
    data["year"] = data['Created At'].map(lambda x: dater(x,1))
    data["month"] = data['Created At'].map(lambda x: dater(x,2))
    data["day"] = data['Created At'].map(lambda x: dater(x,3))
    data["second"] = data['Created At'].map(lambda x: dater(x,4))
    return data[data["year"]==str(year)][['Title','Post Type','Number of Comments','Points','Author']].copy()

# The function returns True if a word is present in stopWordSet
def checkStopWords(word,stopWordSet):
    if word in stopWordSet:
        return True
    else:
        return False
    
# The function creates tokenized titles and adds a new column to the dataframe    
def addTokenizedColumnofTitle(data):
    if 'tokenized_title' in data.columns:
        data = data.drop('tokenized_title', 1)
    data['tokenized_title'] = data['Title'].map(lambda x:tokenizeSentence(x))
    return data

# The function below returns filtered list of tokens given input stopwordSet
def filterByWordList(listOfTokens,stopWordSet):
    filteredList = []
    for token in listOfTokens:
        if checkStopWords(token,stopWordSet):
            continue            
        filteredList.append(token)
    return filteredList

# The function below returns edits tokenized title column of dataframe by filtering stopwords.
def filterTokensByWordList(data,stopWordList):
    stopWordSet = set(stopWordList)
    if 'tokenized_title' in data.columns:
        data['tokenized_title'] = data['tokenized_title'].map(lambda x:filterByWordList(x,stopWordSet))
    else:
        return data
    return data

# The function below returns filtered list based on length of token
def filterByWordLength(listOfTokens):
    filteredList = []
    for token in listOfTokens:
        if len(token)<=2 or len(token)>=9:
            continue            
        filteredList.append(token)
    return filteredList

# The function below returns edits tokenized title column of dataframe by filtering token based on their length.
def filterTokensByWordLength(data):
    if 'tokenized_title' in data.columns:
        data['tokenized_title'] = data['tokenized_title'].map(lambda x:filterByWordLength(x))
    else:
        return data
    return data