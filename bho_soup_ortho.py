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

os.listdir
os.chdir("C:/Users/Owner/Desktop/Data Science/Python")
os.mkdir("transcripts_3")


urls =  ['https://www.rev.com/blog/transcripts/barack-obama-2020-60-minutes-interview-transcript',
          'https://www.rev.com/blog/transcripts/barack-obama-campaign-speech-for-joe-biden-transcript-miami-fl-november-2',
           'https://www.rev.com/blog/transcripts/barack-obama-drive-in-rally-speech-transcript-atlanta-ga-november-2',
           'https://www.rev.com/blog/transcripts/joe-biden-barack-obama-campaign-event-speech-transcript-flint-mi-october-31',
           'https://www.rev.com/blog/transcripts/barack-obama-campaign-speech-for-joe-biden-transcript-orlando-october-27',
           'https://www.rev.com/blog/transcripts/barack-obama-florida-rally-speech-transcript-for-joe-biden-october-24',
           'https://www.rev.com/blog/transcripts/barack-obama-campaign-rally-for-joe-biden-kamala-harris-speech-transcript-october-21',
           'https://www.rev.com/blog/transcripts/barack-obama-campaign-roundtable-event-for-joe-biden-kamala-harris-transcript-october-21',
           'https://www.rev.com/blog/transcripts/barack-obama-gives-best-wishes-to-president-trump-and-melania-trump-after-covid-diagnosis',
           'https://www.rev.com/blog/transcripts/barack-obama-2020-dnc-speech-transcript',
           'https://www.rev.com/blog/transcripts/democratic-national-convention-dnc-night-3-transcript'
        ]

def url_to_trns(url):
    '''Returns transcript data specifically from www.rev.com/blog/transcripts?s=barack+obama.'''
    pg= requests.get(url)
    soup = BeautifulSoup(pg.text, 'html.parser')
    all_p = soup.find_all('p')
    text = [p.text for p in all_p]
    print(url, end='\n'*2)
    return text

# Extract text from Soup Objects 
# LIST each transcript
all_trns = [url_to_trns(u) for u in urls]

# Create Index variable for each transcript
trns_num = ["trns"+str(i+1) for i in range(len(all_trns)) ]


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
[' '.join([data.item.partition(')')[2] for data.item in data if data.value.startswith("Barack Obama")]

data['trns1'][:5]
data2 = data[0,5:]

# We are going to change this to key: comedian, value: string format
def combine_text(list_of_text):
    '''Takes a list of text and combines them into one large chunk of text.'''
    combined_text = ' '.join(list_of_text)
    return combined_text

# Combine it!
data_combined = {key: [combine_text(value)] for (key, value) in data.items()}

pd.set_option('max_colwidth',150)

data_df = pd.DataFrame.from_dict(data_combined).transpose()
data_df.columns = ['transcript']
data_df = data_df.sort_index()
data_df

# Let's take a look at the transcript for Ali Wong
data_df.transcript.loc['ali']

bho_df = DataFrame([' '.join([quote.partition(')')[2] for quote in data if quote.startswith("Barack Obama")])]
                   ,columns=['quote']
                  )














import datetime

dates = []  
date_txt = []
for idx, dte in enumerate(all_trns):
    dte = all_trns[idx][0]
    date_txt.append(dte)
    print(dte)
    dte_obj = datetime.datetime.strptime(dte, '%b %d, %Y')
    dates.append(dte_obj)
    all_trns[idx][0] = dte_obj
del (idx, dte, dte_obj)

print(len(all_trns))
        

tst = [quote["quote"].partition(')')[2] for quote in all_quote if quote["quote"].startswith("Barack Obama")]



bho_list = [' '.join([qte.partition(')')[2] for qte in all_trns if qte.startswith("Barack Obama:")])]

bho_list = [' '.join([qte.partition(')')[2] for qte in all_trns if qte.startswith("Barack Obama:")])]



# # Pickle files for later use

# # Make a new directory to hold the text files
# !mkdir transcripts

# for i, c in enumerate(comedians):
#     with open("transcripts/" + c + ".txt", "wb") as file:
#         pickle.dump(transcripts[i], file)


# del (all_trns)
# pg   = requests.get('https://www.rev.com/blog/transcripts/barack-obama-2020-60-minutes-interview-transcript')
# soup = BeautifulSoup(pg.text, 'html.parser')
# all_p = soup.find_all('p')

# all_quote = []
# text = [p.text for p in soup.find_all('p')]

all_quote = []
bho_list = []
def url_to_trns(url):
    print(url, end='\n \n')
    # print(url)
    pg= requests.get(url)
    soup = BeautifulSoup(pg.text, 'html.parser')
    all_p = soup.find_all('p')
    for idx, qte in enumerate(all_p):
       qte = all_p[idx].getText()
       all_quote.append({'index': idx, 'quote': qte})
    # bho_list.append([' '.join([qte["quote"].partition(')')[2] for qte in all_quote if qte["quote"].startswith("Barack Obama")])])
    # bho_list = append([' '.join([quote["quote"].partition(')')[2] for quote in all_quote
                    # if quote["quote"].startswith("Barack Obama")])])
                   
    
    # bho_df = DataFrame([' '.join([quote["quote"].partition(')')[2] 
    #                for quote in all_quote
    #                if quote["quote"].startswith("Barack Obama")])]
    #                ,columns=['quote']
    #               )
    
                  
            
    # bho_qte = [' '.join([quote["quote"].partition(')')[2] for quote in all_quote if quote["quote"].startswith("Barack Obama")])]                            ,columns=['quote']
    # text = [p.text for p in soup.find_all('p')]
    return all_quote, bho_df, bho_list
    

all_trns = [url_to_trns(u) for u in urls]


    
    
    
    
    
    
    
   
import datetime

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
