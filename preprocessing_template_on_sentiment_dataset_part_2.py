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
pd.set_option('display.max_colwidth', 100)

kainourgio = pd.read_csv("dataset_with_POS")

################################################
####SECOND PART OF PREPROCESSING################
################################################


kainourgio = pd.read_csv("dataset_with_POS")

pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_columns', None)

kainourgio['OriginalTweet'] = kainourgio['OriginalTweet'].str.lower()

kainourgio.drop(['Location','TweetAt'],axis=1,inplace = True)

#vgazoume ta links
kainourgio['OriginalTweet'] = kainourgio['OriginalTweet'].replace(r'http\S+|www\S+|https\S+', ' ', regex=True)
# kanoume replace ta usernames
kainourgio['OriginalTweet'] = kainourgio['OriginalTweet'].replace(r'@([A-Za-z0-9_]+)', ' ', regex=True)
#kanoume replace ta hashtags
kainourgio['OriginalTweet'] = kainourgio['OriginalTweet'].replace(r'#([A-Za-z0-9_]+)', ' ', regex=True)
#removal of more of two consequtive characters

kainourgio['OriginalTweet'] = kainourgio['OriginalTweet'].str.replace(r'(.)\1+', r'\1\1')
#kanoume replace to &amp pou vrisketai synxa
kainourgio['OriginalTweet'] = kainourgio['OriginalTweet'].replace('&amp', ' ', regex=True)
#re.sub('[^a-zA-Z]', ' ', kainourgio['OriginalTweet'][34])
#vazoume ena keno sta ellinika erwtimatika i anw teleia stin agglika glwssa
kainourgio['OriginalTweet'] = kainourgio['OriginalTweet'].replace(';', '', regex=True)

kainourgio['OriginalTweet'] = kainourgio['OriginalTweet'].replace('-', ' ', regex=True)
kainourgio['OriginalTweet'] = kainourgio['OriginalTweet'].replace('[?]|[-]|[!]', ' ', regex=True)
kainourgio['OriginalTweet'] = kainourgio['OriginalTweet'].replace('[.]', ' . ', regex=True)

#afairoume ta mi aparaitita kena
kainourgio['OriginalTweet'] = kainourgio['OriginalTweet'].replace('\s+', ' ', regex=True)
#diagrafoume toys mi asci xaraktires
kainourgio['OriginalTweet'] = kainourgio['OriginalTweet'].str.encode('ascii', 'ignore').str.decode('ascii')

#removal of more of two consequtive characters

#eisagwgi twn stopwords kai anadiorganwsi tou lathos tou spacy pou den eixe swsta stopwords list

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
		
	#vgazoume simia stixis kai kena symbola kai lexeis panw apo duo sumvola   
	string_two = [token.text for token in string_one if not token.is_punct 
			   | token.is_space 
			   | (token.pos_ == 'PUNCT') 
			   | (token.pos_=="X") 
			   | (len(token.text)<=2) 
			   | (token.pos_ == "NUM")]
	string_three = ' '.join([str(elem) for elem in string_two])
	string_four = nlp(string_three)
	#dimiourgia lemmatization
	string_five = [token.lemma_ for token in string_four if not ((token.lemma_ == "-PRON-") | (token.pos_ == "NUM"))]
	
	string_six = [token for token in string_five if not token in nlp.Defaults.stop_words]
	string_seven = ' '.join([str(elem) for elem in string_six])
	string_eight = nlp(string_seven)
	string_teliko = [token.norm_ for token in string_eight]
	return string_teliko


#gia mia timi	
string_new = diadikasia_preprocessing_new(kainourgio['OriginalTweet'][3])

for i in teliko_new.index:
	teliko_new.at[i,"OriginalTweet"]=diadikasia_preprocessing_new(kainourgio['OriginalTweet'][i])
	print("lexi pou epexergazomaste einai h:",i )


for i in teliko_new.index:
	print(teliko_new['OriginalTweet'][i])
	



teliko_new.to_csv('dataset_preprocessed_teliko',index=False)
teliko_new = pd.read_csv("dataset_final")
#an exoume thema sto inport kai anti gia listes pairnoume string
teliko_new["OriginalTweet"] =teliko_new["OriginalTweet"].apply(literal_eval)

test_kommati = teliko_new
test_kommati['OriginalTweet'][0]

#ftiaxnoume mia megali lista me oles tis lexeis
string_list =[]
for i in test_kommati.index:
	for j in range(0,len(test_kommati["OriginalTweet"][i])):
		string_list.append(test_kommati["OriginalTweet"][i][j])
        
#sunartisi gia na upologisoume tis ksexwristes times se mia lista	
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
most_occur = Counter.most_common(8000) 
least_common = Counter.most_common()[-3:]
type(most_occur)
#upologizoume tis lexeis pou emfanizontai pio syxna me skopo na tis emfanizoume swsta
keys = []
values = []
for k,v in most_occur:
	keys.append(k) 
	values.append(v) 
plt.figure(figsize=(20,10))
plt.xticks(rotation=90)
sns.barplot(x=keys,y=values)
xlabel = ("lexeis")
ylabel = ("arithmos emfanisewn")	

#emfanizoume tis pio sunxes lexeis
for k,v in most_occur:
	print(f"{k:{10}} {v:>{10}}")
	
teliko_new.to_csv('dataset_final',index=False)

#sunartisi gia na doume an ena string exei mesa noumera
def num_there(s):
    return any(i.isdigit() for i in s)	


most_common_15k = keys
#diagrafoume oses lexeis exoun akoma simeia stixis i arithmous
for i in teliko_new.index:
	teliko_new["OriginalTweet"][i] = [x for x in teliko_new["OriginalTweet"][i] if not (num_there(x)) | (x == "covid")]
for i in teliko_new.index:	
	teliko_new["OriginalTweet"][i] = [x for x in teliko_new["OriginalTweet"][i] if bool(re.match('^[a-zA-Z]*$',x))]
for i in teliko_new.index:	
	teliko_new["OriginalTweet"][i] = [x for x in teliko_new["OriginalTweet"][i] if len(x)>=3]
for i in teliko_new.index:	
	teliko_new["OriginalTweet"][i] = [x for x in teliko_new["OriginalTweet"][i] if not x == "covid"]	

for i in teliko_new.index:	
	teliko_new["OriginalTweet"][i] = [x for x in teliko_new["OriginalTweet"][i] if  x in most_common_15k]

for i in test_kommati.index:
	print("epexergazomaste to tweet se index:",i)	
	test_kommati["OriginalTweet"][i] = [x for x in test_kommati["OriginalTweet"][i] if  x in keys]


from english_words import english_words_set


test_kommati.to_csv('dataset_8K_uniq_words',index=False)
#sunartisi gia allagi ta -1 se 0 wste na exoume omoiomorfes katigories
test_kommati.to_csv('dataset_15K_uniq_words',index=False)


teliko_new = pd.read_csv("dataset_10K_uniq_words")
teliko_new["OriginalTweet"] =teliko_new["OriginalTweet"].apply(literal_eval)

teliko_new["OriginalTweet"][0]        













