# -*- coding: utf-8 -*-
"""
PREPROCESSING TEMPLATE PART 2

"""




import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from collections import Counter
from ast import literal_eval

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 10000)

kainourgio = pd.read_csv("got_users_with_POS.csv")

################################################
####SECOND PART OF PREPROCESSING################
################################################



pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_columns', None)

kainourgio['tweets'] = kainourgio['tweets'].str.lower()
kainourgio['tweets'] = kainourgio['tweets'].replace(r'\<u\+2019>', "'", regex=True)

kainourgio['tweets'] = kainourgio['tweets'].replace(r'<u+(.*?)>', ' ', regex=True)
    

#removing links
kainourgio['tweets'] = kainourgio['tweets'].replace(r'http\S+|www\S+|https\S+', ' ', regex=True)
# replace usernames
kainourgio['tweets'] = kainourgio['tweets'].replace(r'@([A-Za-z0-9_]+)', ' ', regex=True)
# replace  hashtags
kainourgio['tweets'] = kainourgio['tweets'].replace(r'#([A-Za-z0-9_]+)', ' ', regex=True)
#removal of more of two consequtive characters

kainourgio['tweets'] = kainourgio['tweets'].str.replace(r'(.)\1+', r'\1\1')
#replace &amp word
kainourgio['tweets'] = kainourgio['tweets'].replace('&amp', ' ', regex=True)
#removing specific characters, these characters can be removed properly using regex 
kainourgio['tweets'] = kainourgio['tweets'].replace(';', '', regex=True)

kainourgio['tweets'] = kainourgio['tweets'].replace('-', ' ', regex=True)
kainourgio['tweets'] = kainourgio['tweets'].replace('[?]|[-]|[!]', ' ', regex=True)
kainourgio['tweets'] = kainourgio['tweets'].replace('[.]', ' . ', regex=True)

#removing gaps
kainourgio['tweets'] = kainourgio['tweets'].replace('\s+', ' ', regex=True)
#removing ASCI characters
kainourgio['tweets'] = kainourgio['tweets'].str.encode('ascii', 'ignore').str.decode('ascii')


#we add new stop words 

import spacy
nlp = spacy.load('en_core_web_lg')

nlp = spacy.load('en_core_web_lg')

for word in nlp.Defaults.stop_words:
    lex = nlp.vocab[word]
    lex.is_stop = True

#adding stopword to list
nlp.Defaults.stop_words.add('amp')
nlp.vocab['amp'].is_stop = True

# Remove the word from the set of stop words
nlp.Defaults.stop_words.remove('not')
nlp.vocab['not'].is_stop = False

nlp.Defaults.stop_words.add('covid-19')
nlp.vocab['covid-19'].is_stop = True

# checking if a word is in stopword list
nlp.vocab['go'].is_stop


teliko_new = kainourgio


def diadikasia_preprocessing_new(string):
	string_one = nlp(string)
	#normalize tis lexeis dld to n't tha ginei not
	#string_two = [token.norm_ for token in string_one]
		
	
	#we remove punctuation with spacy and words with two or less characters   
	string_two = [token.text for token in string_one if not token.is_punct 
			   | token.is_space 
			   | (token.pos_ == 'PUNCT') 
			   | (token.pos_=="X") 
			   | (len(token.text)<=2) 
			   | (token.pos_ == "NUM")]
	string_three = ' '.join([str(elem) for elem in string_two])
	string_four = nlp(string_three)
	#we execute lemmatization
	string_five = [token.lemma_ for token in string_four if not ((token.lemma_ == "-PRON-") | (token.pos_ == "NUM"))]
	
	string_six = [token for token in string_five if not token in nlp.Defaults.stop_words]
	string_seven = ' '.join([str(elem) for elem in string_six])
	string_eight = nlp(string_seven)
	string_teliko = [token.norm_ for token in string_eight]
	return string_teliko

	
string_new = diadikasia_preprocessing_new(kainourgio['OriginalTweet'][3])
teliko_new.index

teliko_new.to_csv('got_with_POS_pre_2_1.csv',index=False)
teliko_new = pd.read_csv('got_with_POS_pre_2_3.csv')

teliko_new["tweets"]=teliko_new["tweets"].apply(literal_eval)

 
test = teliko_new[:300] 

# we visualize some data  
x=teliko_new['ADJ'].index.values
y=	teliko_new['ADJ'].values
plt.scatter(x,y,alpha = 0.5,marker='.',label='ADJ')
plt.xticks(rotation=90)

x=teliko_new['ADP'].index.values
y=	teliko_new['ADP'].values
plt.scatter(x,y,alpha = 0.5,marker='*',label='ADP')
plt.xticks(rotation=90)

x=teliko_new['PUNCT'].index.values
y=	teliko_new['PUNCT'].values
plt.scatter(x,y,alpha = 0.5,marker='x',label='PUNCT')
plt.xticks(rotation=90)
plt.legend()


#we apply this function to deal with import issuses, list of strings should work properly afeter this
teliko_new["tweets"]=teliko_new["tweets"].apply(literal_eval)

test_kommati = teliko_new

#we create a list with all the words of the users
string_list =[]
for i in test_kommati.index:
	for j in range(0,len(test_kommati["tweets"][i])):
		string_list.append(test_kommati["tweets"][i][j])
        
#function to find the unique values	
def unique(list1):   
    # intilize a null list 
    unique_list = []       
    # traverse for all elements 
    for x in list1: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x) 
    return unique_list	
 
 
unique_list = unique(string_list)	

Counter = Counter(string_list)
most_occur = Counter.most_common(60000)
most_occur.keys 
least_common = Counter.most_common()[-100:]
type(most_occur)
#we find the most common words
keys = []
values = []
for k,v in most_occur[:1000]:
	keys.append(k) 
	values.append(v) 
keys[:100]
values[:100]    
plt.figure(figsize=(20,10))
plt.xticks(rotation=90)
sns.barplot(x=keys[:100],y=values[:100] )
xlabel = ("lexeis")
ylabel = ("arithmos emfanisewn")	
plt.savefig("most_occur.png", dpi=2000)
#printing most common words
for k,v in most_occur:
    print(k,v)
    print(f"{k:{10}} {v:>{10}}")
	

#function to check if there is a number in a word
def num_there(s):
    return any(i.isdigit() for i in s)	


most_common_15k = keys
#removing words with punctuation and numbers that continues to exist in the words
for i in teliko_new.index:
    teliko_new["tweets"][i] = [x for x in teliko_new["tweets"][i] if   bool(re.match('^[a-zA-Z]*$',x))]
    print(i,"grammata")

teliko_new.to_csv('got_with_POS_pre_2_2.csv',index=False)
   
for i in teliko_new.index:
    teliko_new["tweets"][i] = [x for x in teliko_new["tweets"][i] if len(x)>=3 ]
    print(i,"arithmos lexewn")

teliko_new.to_csv('got_with_POS_pre_2_3.csv',index=False)

# make dataset with specific vocabulary
for i in teliko_new.index:	
	teliko_new["tweets"][i] = [x for x in teliko_new["tweets"][i] if  x in most_common_15k]

for i in test_kommati.index:
	print("epexergazomaste to tweet se index:",i)	
	test_kommati["tweets"][i] = [x for x in test_kommati["tweets"][i] if  x in keys]




test_kommati.to_csv('dataset_8K_uniq_words',index=False)

test_kommati.to_csv('dataset_15K_uniq_words',index=False)


teliko_new = pd.read_csv("dataset_10K_uniq_words")
teliko_new["OriginalTweet"] =teliko_new["OriginalTweet"].apply(literal_eval)














