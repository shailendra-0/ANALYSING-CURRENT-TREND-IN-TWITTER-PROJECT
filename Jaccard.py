import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
import tweepy
import configparser
import pandas
import json
import re

import tweepy
import csv
import os
import pandas as pd

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import re
#import spacy
#from sklearn.model_selection import train_test_split
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.tokenize import RegexpTokenizer, WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from string import punctuation
import collections
from collections import Counter
#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#import en_core_web_sm

#from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#!pip3 install -U spacy
#!python3 -m spacy download en_core_web_sm

#from sklearn.metrics import jaccard_score

file=open("economy.txt")
economy_related_words = file.read()
file.close()

file=open("social.txt")
social_related_words = file.read()
file.close()

file=open("culture.txt")
culture_related_words  = file.read()
file.close()

file=open("health.txt")
health_related_words =file.read()
file.close()

def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)


def get_scores(group,tweets):
    scores = []
    for tweet in tweets:
        s = jaccard_similarity(group, tweet)
        scores.append(s)
    return scores

economy_list=0
a=0
social_list=0
b=0
culture_list=0
c=0
health_list=0
d=0

token_economy=economy_related_words.split()
token_social=social_related_words.split()
token_culture=culture_related_words.split()
token_health=health_related_words.split()

file=open('a.text','r')
lines=file.readlines()

c=0

for line in lines:
    c+=1
    s=line.strip()

    s=s.lower()
    s=re.sub('['+punctuation+']+','',s)
    s=re.sub('([0-9]+)','',s)


    s=re.sub(r'[^\w\s]','',s)
    s=re.sub(r'http\S+','',s)
    s=re.sub(r'bit.ly/\S+','',s)
    s=re.sub(r'http\S+','',s)
    s=s.strip('[link]')

    s=re.sub('(RT[A-Za-z]+[A-Za-z0-9-_]+)','',s)
    s=re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)','',s)
    s=re.sub('(#[A-Za-z]+[A-Za-z0-9-_]+)','',s)
    
    #print(s)

    s=s.split()

    a=jaccard_similarity(token_economy,s)
    b=jaccard_similarity(token_social,s)
    c=jaccard_similarity(token_culture,s)
    d=jaccard_similarity(token_health,s)

    if a!=0 or b!=0 or c!=0 or d!=0:
        if a>b and a>c and a>d:
            economy_list=economy_list+1
        if b>a and b>c and b>d:
            social_list=social_list+1
        if c>a and c>b and c>d:
            culture_list=culture_list+1
        if d>a and d>b and d>c:
            health_list=health_list+1

file.close()

s1="hello world".split()
s2="hello a b world".split()
#print(s1)

e=jaccard_similarity(s1,s2)
print("Jaccard Similarity")
#print("\nJaccard similarity:",e)

ep=economy_list/(economy_list+social_list+culture_list+health_list)
ep=ep*100
sp=social_list/(economy_list+social_list+culture_list+health_list)
sp=sp*100
cp=culture_list/(economy_list+social_list+culture_list+health_list)
cp=cp*100
hp=health_list/(economy_list+social_list+culture_list+health_list)
hp=hp*100



print("\nPercentage: ")

print(f"Economy: {ep} %")
print(f"Social: {sp} %")
print(f"Culture: {cp} %")
print(f"Health: {hp} %")

import numpy as np
import matplotlib.pyplot as plt
 
cars = ['Economy','Social','Culture','Health']
data = [economy_list,social_list,culture_list,health_list]
 
 

explode = (0.1, 0.0, 0.2, 0.3)
 

colors = ( "orange", "cyan", "brown",
           "grey")
 

wp = { 'linewidth' : 1, 'edgecolor' : "green" }
 

def func(pct, allvalues):
     absolute = int(pct / 100.*np.sum(allvalues))
     return "{:.1f}%\n({:d} Tweets)".format(pct, absolute)
 

fig, ax = plt.subplots(figsize =(10, 7))
wedges, texts, autotexts = ax.pie(data,autopct = lambda pct: func(pct, data),explode = explode,labels = cars,shadow = True,colors = colors,startangle = 90,wedgeprops = wp,textprops = dict(color ="magenta"))
 

ax.legend(wedges, cars,
           title ="Trending Topic",
           loc ="center left",
           bbox_to_anchor =(1, 0, 0.5, 1))
 
plt.setp(autotexts, size = 8, weight ="bold")
ax.set_title("Trend Analysis")
 
plt.show()



