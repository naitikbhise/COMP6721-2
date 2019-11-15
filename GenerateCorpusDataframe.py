import pandas as pd
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
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

def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']

def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']

def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']

def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return None

def sentence_split(sentence):
    text = nltk.word_tokenize(sentence)
    pos_text = nltk.pos_tag(text)
    new_text = []
    grab = False
    lemmatizedWord = None
    for word,pos in pos_text:
        #print(word)
        #print(pos)
        tag = penn_to_wn(pos)
        if pos == 'NNP' and grab == True:
            grab = False
            #print(word)
            lemmatizedWord += " " + wordnet_lemmatizer.lemmatize(word,tag)
            new_text.append(lemmatizedWord.lower())
            lemmatizedWord = None
        elif pos == 'NNP' and grab == False:
            lemmatizedWord = wordnet_lemmatizer.lemmatize(word,tag)
            grab = True
            #print(grab)
        elif pos != 'NNP':
            if lemmatizedWord:
                if lemmatizedWord.isalpha():
                    new_text.append(lemmatizedWord.lower())
            if tag is None:
                lemmatizedWord = word
            else:
                lemmatizedWord = wordnet_lemmatizer.lemmatize(word,tag)
            if lemmatizedWord.isalpha() or pos=='JJ':
                if word.isalpha():
                    new_text.append(lemmatizedWord.lower())
            lemmatizedWord = None
            grab = False
    return new_text

def addTokenizedColumnofTitle(data):
    data['tokenized_title'] = data['Title'].map(lambda x:sentence_split(x))
    return data

