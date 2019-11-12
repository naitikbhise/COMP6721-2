import pandas as pd
import nltk

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

def addTokenizedColumnofTitle(data):
    data['tokenized_title'] = data['Title'].map(lambda x:nltk.word_tokenize(x.lower()))
    return data

