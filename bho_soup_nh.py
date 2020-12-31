# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 11:49:38 2020

@author: Owner
"""
#### SEARCH LANDING PAGE ####
# https://www.rev.com/blog/transcripts?s=barack+obama
    
# URL = 'https://joebiden.com/racial-economic-equity/#'
# page = requests.get(URL)

# URL = 'https://joebiden.com/racial-economic-equity/#'
    ## https://www.nber.org/papers/w27462
    ## https://www.nber.org/system/files/working_papers/w27462/w27462.pdf

# bid_har = requests.get('https://joebiden.com/racial-economic-equity/#')


# 
########

import requests   #allows us to download initial HTML info
from bs4 import BeautifulSoup  # allows for usage of DATA within HTML
import pandas as pd
from pandas import DataFrame
import re
import pickle

import string
from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
import nltk
nltk.download('averaged_perceptron_tagger')
stopwords = stopwords.words('english')
print(stopwords)

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

# # PICKLE files
# Create a key value for each transcript
# trns_num = ["trns"+str(i+1) for i in range(len(urls)) ]
# for i,n in enumerate(trns_num):
#     with open("transcripts/" + n + ".txt", "wb") as file:
#         pickle.dump(all_trns[i], file)

# dict_keys
# LOAD pickled files
data = {}
for i, c in enumerate(trns_num):
    with open("transcripts/" + c + ".txt", "rb") as file:
        data[c] = pickle.load(file)

import datetime
print(data.keys())

dates = []  
for k in data.keys():
    dte = data[k][0]    
    dte_obj = datetime.datetime.strptime(dte, '%b %d, %Y').date()
    print(k,dte,dte_obj)
    dates.append(dte_obj)
     # data=dict({zip((k,dte_obj):data[k][1:]})
del (k, dte, dte_obj)




print(data["trns2"][0], type(data["trns2"][0]))


dict((d1[key], value) for (key, value) in d.items())



all_quote = []
def url_to_trns(url):
    print(url, end='\n \n')
    pg= requests.get(url)
    soup = BeautifulSoup(pg.text, 'html.parser')
    all_p = soup.find_all('p')
    
    all_quote = []
    for idx, qte in enumerate(all_p):
       qte = all_p[idx].getText()       
       all_quote.append({'index': idx, 'quote': qte})
        ###
        #Intermediate step to create list where date is first element
        ###
        
    text=[' '.join([quote["quote"].partition(')')[2] 
                    for quote in all_quote
                    if quote["quote"].startswith("Barack Obama")])]
    return  text

all_trns_df = DataFrame([url_to_trns(u) for u in urls]
                       ,columns = ["transcript"])
all_trns_df.Index(trns_num)

os.chdir('C:/Users/Owner/Desktop/Data Science/Python')



all_trns_df = DataFrame(data, columns = "transcript")

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
                      , min_df=4
                      , ngram_range=(4,6)
                     )
vectorizer = cv.fit_transform(all_trns_df.trns_mod)
vectors = pd.DataFrame(vectorizer.toarray(), columns = cv.get_feature_names())
print (vectors)
    
po = vectors["poor"]

    
    
    
    
    
    
    
   



