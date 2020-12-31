# -*- coding: utf-8 -*-
"""
This is an experiment in processing natural language from political 
interviews and speeches.  Transcripts will be scraped the web using 
Requests and BeautifulSoup modules and processed using SciKitLearn module.
After EDA, topic modeling and sentiment analysis will be performed with an 
additional goal of using Recursive NNs for sentence-sentiment analysis.
"""
#### SEARCH LANDING PAGE ####
# https://www.rev.com/blog/transcripts?s=barack+obama
    
        # URL = 'https://joebiden.com/racial-economic-equity/#'
        # page = requests.get(URL)
        
        # URL = 'https://joebiden.com/racial-economic-equity/#'
            ## https://www.nber.org/papers/w27462
            ## https://www.nber.org/system/files/working_papers/w27462/w27462.pdf
        
        # bid_har = requests.get('https://joebiden.com/racial-economic-equity/#')


#### Modules
import requests   #allows us to download initial HTML info
from bs4 import BeautifulSoup  # allows for usage of DATA within HTML

import pandas as pd
from pandas import DataFrame
import re
import pickle
import os

import string
from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
import nltk
nltk.download('averaged_perceptron_tagger')
stopwords = stopwords.words('english')
print(stopwords)

os.chdir('C:/Users/Owner/Desktop/Data Science/Python')
os.getcwd()

### OBTAIN robot.txt 
req = 'https://www.rev.com/robots.txt'
rev_bot = requests.get(req)
bot_soup = BeautifulSoup(rev_bot.text,'html.parser')
print(bot_soup.prettify())


### LIST URLs of Transcripts
urls =  ['https://www.rev.com/blog/transcripts/barack-obama-2020-60-minutes-interview-transcript'
         ,'https://www.rev.com/blog/transcripts/barack-obama-campaign-speech-for-joe-biden-transcript-miami-fl-november-2'
         ,'https://www.rev.com/blog/transcripts/barack-obama-drive-in-rally-speech-transcript-atlanta-ga-november-2'
         # ,'https://www.rev.com/blog/transcripts/joe-biden-barack-obama-campaign-event-speech-transcript-flint-mi-october-31'
         # ,'https://www.rev.com/blog/transcripts/barack-obama-campaign-speech-for-joe-biden-transcript-orlando-october-27'
         ,'https://www.rev.com/blog/transcripts/barack-obama-florida-rally-speech-transcript-for-joe-biden-october-24'
         # ,'https://www.rev.com/blog/transcripts/barack-obama-campaign-rally-for-joe-biden-kamala-harris-speech-transcript-october-21'
         ,'https://www.rev.com/blog/transcripts/barack-obama-campaign-roundtable-event-for-joe-biden-kamala-harris-transcript-october-21'
         ,'https://www.rev.com/blog/transcripts/barack-obama-gives-best-wishes-to-president-trump-and-melania-trump-after-covid-diagnosis'
         ,'https://www.rev.com/blog/transcripts/barack-obama-2020-dnc-speech-transcript'
         ,'https://www.rev.com/blog/transcripts/democratic-national-convention-dnc-night-3-transcript'
        ]


all_quote = []
def url_to_trns(url):
    print(url, end='\n'*2)
    
    pg= requests.get(url)
    soup = BeautifulSoup(pg.text, 'html.parser')
    all_p = soup.find_all('p')
    
    all_quote = []
    for idx, qte in enumerate(all_p):
       qte = all_p[idx].getText()       
       all_quote.append({'index': idx, 'quote': qte})
        
    text=[' '.join([quote["quote"].partition(')')[2] 
                    for quote in all_quote
                    if quote["quote"].startswith("Barack Obama")])]
    return  text


# Create Index variable for each transcript
trns_num = ["trns"+str(i+1) for i in range(len(urls)) ]


all_trns_df = DataFrame([url_to_trns(u) for u in urls]
                       ,columns = ["transcript"]
                       ,index = trns_num)

#PICKLE each transcript
for i,n in enumerate(trns_num):
    with open("transcripts/" + n + ".txt", "wb") as file:
        pickle.dump(all_trns_df.iloc[i], file)


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

all_trns_df['trns_mod'] = all_trns_df['transcript'].apply(lambda x: clean_string(x))
all_trns_df['wc_0']     = all_trns_df['transcript'].apply(lambda x: word_count_WC(x))
all_trns_df['wc_mod']   = all_trns_df['trns_mod'].apply(lambda x: word_count_WC(x))


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words = 'english'
                      , min_df=1
                      , ngram_range=(1,5)
                     )
vectorizer = cv.fit_transform(all_trns_df.trns_mod)
vectors = pd.DataFrame(vectorizer.toarray()
                       ,columns = cv.get_feature_names()
                       ,index = trns_num
                       )
print (vectors)
    
po = vectors["poor"]



   
dates = []  
for idx, dte in enumerate(all_trns):
    dte = all_trns[idx][0]
    print(dte)
    dte_obj = datetime.datetime.strptime(dte, '%b %d, %Y')
    dates.append(dte_obj)
del (idx, dte, dte_obj)

dates = []  
for idx, dte in enumerate(all_trns):
    dte = all_trns[idx][0]
    print(dte)
    dte_obj = datetime.datetime.strptime(dte, '%b %d, %Y')
    dates.append(dte_obj)
    all_trns[idx][0] = dte_obj
del (idx, dte, dte_obj)
 
    
    
    
    
    
    
   



