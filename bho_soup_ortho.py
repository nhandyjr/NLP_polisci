# -*- coding: utf-8 -*-
"""
This is an experiment in processing natural language from political 
interviews and speeches.  Transcripts will be scraped the web using 
Requests and BeautifulSoup modules and processed using SciKitLearn module.
After EDA, topic modeling and sentiment analysis will be performed with an 
additional goal of using Recursive NNs for sentence-sentiment analysis.
"""
### SEARCH LANDING PAGE ####
# https://www.rev.com/blog/transcripts?s=barack+obama
    
# 
########



import requests   #allows download of initial HTML info
from bs4 import BeautifulSoup  # allows parsing of DATA within HTML

import pandas as pd
from pandas import DataFrame

import re
import os
import pickle
import string
import string
from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
import nltk
nltk.download('averaged_perceptron_tagger')
stopwords = stopwords.words('english')
print(stopwords)

os.listdir
os.chdir("C:/Users/Owner/Desktop/Data Science/Python")
# os.mkdir("transcripts_3")


urls =  ['https://www.rev.com/blog/transcripts/barack-obama-2020-60-minutes-interview-transcript',
          'https://www.rev.com/blog/transcripts/barack-obama-campaign-speech-for-joe-biden-transcript-miami-fl-november-2',
           'https://www.rev.com/blog/transcripts/barack-obama-drive-in-rally-speech-transcript-atlanta-ga-november-2',
           # 'https://www.rev.com/blog/transcripts/joe-biden-barack-obama-campaign-event-speech-transcript-flint-mi-october-31',
           'https://www.rev.com/blog/transcripts/barack-obama-campaign-speech-for-joe-biden-transcript-orlando-october-27',
           'https://www.rev.com/blog/transcripts/barack-obama-florida-rally-speech-transcript-for-joe-biden-october-24',
           'https://www.rev.com/blog/transcripts/barack-obama-campaign-rally-for-joe-biden-kamala-harris-speech-transcript-october-21',
           'https://www.rev.com/blog/transcripts/barack-obama-campaign-roundtable-event-for-joe-biden-kamala-harris-transcript-october-21',
           'https://www.rev.com/blog/transcripts/barack-obama-gives-best-wishes-to-president-trump-and-melania-trump-after-covid-diagnosis',
           'https://www.rev.com/blog/transcripts/barack-obama-2020-dnc-speech-transcript',
           'https://www.rev.com/blog/transcripts/democratic-national-convention-dnc-night-3-transcript'
        ]


def url_to_trns(url):
    '''Returns transcript data specifically from www.rev.com/blog/transcripts?s=barack+obama.
       where Barack Obama is the speaker
    '''
    pg= requests.get(url)
    soup = BeautifulSoup(pg.text, 'html.parser')
    all_p = soup.find_all('p')
    print(url, end='\n'*2)   
    
    all_quote = []
    for idx, qte in enumerate(all_p):
       qte = all_p[idx].getText()       
       all_quote.append({'index': idx, 'quote': qte})
        
    text=[' '.join([quote["quote"].partition(')')[2] 
                    for quote in all_quote
                    if quote["quote"].startswith("Barack Obama")])]
    return text


#### Extract text from Soup Objects 
#### LIST each transcript
all_trns = [url_to_trns(u) for u in urls]

# Create Index variable for each transcript
trns_num = ["trns"+str(i+1) for i in range(len(urls)) ]


# PICKLE files and index each transcript 
for i,n in enumerate(trns_num):
    with open("transcripts_3/" + n + ".txt", "wb") as file:
        pickle.dump(all_trns[i], file)

# LOAD pickled files
data = {}
for i, c in enumerate(trns_num):
    with open("transcripts_3/" + c + ".txt", "rb") as file:
        data[c] = pickle.load(file)
          
print(data.keys())


pd.set_option('max_colwidth',150)

data_df = pd.DataFrame.from_dict(data).transpose()
data_df.columns = ["quotes"]

def clean_string(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = ''.join([word for word in text if word not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stopwords])    
    return text

def word_count_WC(x):
    my_astring = x.lower().split()
    sum = 0
    for item in my_astring:
        sum = sum + 1
    return sum


data_df['quotes_mod'] = data_df['quotes'].apply(lambda x: clean_string(x))
data_df['wc_0']     = data_df['quotes'].apply(lambda x: word_count_WC(x))
data_df['wc_mod']   = data_df['quotes_mod'].apply(lambda x: word_count_WC(x))


#### Document-Term Matrix
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words = 'english'
                      , min_df=2
                      , ngram_range=(1,6)
                     )

vectorizer = cv.fit_transform(data_df.quotes_mod)

vectors = pd.DataFrame(vectorizer.toarray()
                       ,columns = cv.get_feature_names()
                       ,index = trns_num
                       )
print (vectors)


'''
EXPLORATORY DATA ANALYSIS
(1) Top Words
(2) Vocabulary: Unique number of words used
    * might need to normalize a time-frame for each transcripts
(3) Top 4-n-grams
'''


'''
SENTIMENT ANALYSIS

'''

# pip install textblob

from textblob import TextBlob as tb

pol = lambda x: tb(x).sentiment.polarity
sub = lambda x: tb(x).sentiment.subjectivity

data_df['polarity'] = data_df['quotes_mod'].apply(pol)
data_df['subjectivity'] = data_df['quotes_mod'].apply(sub)

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10,8]

for index, trns in enumerate(data_df.index):
    x = data_df.polarity.loc[trns]
    y = data_df.subjectivity.loc[trns]
    plt.scatter(x,y,color='blue')
    plt.text(x+.001, y+.001,trns)
    plt.xlim(0,.5)
    plt.ylim(.25,.5)

plt.title('Sentiment Analysis', fontsize=20)
plt.xlabel('<--Negative------- Positive -->', fontsize=15)
plt.ylabel('<--Facts------- Opinion-->', fontsize=15)

plt.show()


