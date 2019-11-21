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
    nltk.data.find('corpora/treebank')
except:
    nltk.download('treebank')
try:
    nltk.data.find('corpora/brown')
except:
    nltk.download('brown')
try:
    nltk.data.find('corpora/conll2000')
except:
    nltk.download('conll2000')
try:
    nltk.data.find('corpora/conll2002')
except:
    nltk.download('conll2002')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except:
    nltk.download('averaged_perceptron_tagger')

from nltk.corpus import wordnet as wn
wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()

from nltk.corpus import treebank, conll2000, brown, conll2002
from nltk import DefaultTagger, UnigramTagger, BigramTagger
train_sents = treebank.tagged_sents() + brown.tagged_sents() + conll2000.tagged_sents() + conll2002.tagged_sents()
edited_train = []
for sent in train_sents:
    edited_train.append([(word.lower(),tag) for (word,tag) in sent])
t0 = DefaultTagger(None)
et1 = UnigramTagger(edited_train, backoff = t0)
et2 = BigramTagger(edited_train, backoff = et1)

def penn_to_wn(tag):
    nltk_wn_pos = {'J':wn.ADJ,'V':wn.VERB,'N':wn.NOUN,'R':wn.ADV}
    try:
        return nltk_wn_pos[tag[0]]
    except:
        return None
    

def skipUnwantedTokens(token):
    punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/','”','“','–',
                ':', ';', '<', '=', '>', '?', '@', '[',  ']', '^', '_', '`', '{', '|','}','~','’']
    if token in punctuations:
        return True
    else:
        return False

def filterUnwantedCharacters(tokenList):
    NewList = []
    for token in tokenList:
        if token in ["'s","'-","''",'\\',]:
            continue
        editedToken = ''
        for char in token:
            if skipUnwantedTokens(char):
                continue
            editedToken += char
        if len(editedToken):
            NewList.append(editedToken)
    return NewList

def tokenizeSentence(sentence):
    #sentence = sentence.lower()
    tokenized = []
    text = nltk.word_tokenize(sentence)
    text = filterUnwantedCharacters(text)
    grab = False
    #for word,pos in et2.tag(text):
    for word,pos in nltk.pos_tag(text):
        tag = penn_to_wn(pos)
        if tag is None:
            lemmatizedWord = word
        else:
            lemmatizedWord = wordnet_lemmatizer.lemmatize(word,tag)
        lemmatizedWord = lemmatizedWord.lower()
        #if pos=='NN' or pos=='NNS':
        if pos=='NNP':    
            if grab:
                tokenized[-1] += " " + lemmatizedWord
                grab = False
                continue
            else:
                grab = True
        else:
            if grab:
                grab = False
        tokenized.append(lemmatizedWord)
    return tokenized
    

