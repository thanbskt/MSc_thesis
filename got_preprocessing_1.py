# -*- coding: utf-8 -*-
"""
PREPROCESSING TEMPLATE PART 1

"""
#importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from collections import Counter
from ast import literal_eval

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)

#kanoume import to dataset

train = pd.read_csv("users_with_text")
train = train[['user_id','tweets','user_favorites','listed_count','statuses_count','verified','final_cred_score']]
#train = pd.read_csv("Corona_NLP_train.csv", encoding= 'ISO-8859-1')

###############################################
####FIRST PART OF PREPROCESSING################
###############################################
train.info()
train.describe()
#ploting  sentiment column and we get 5 different classes

sns.distplot(train['final_cred_score'],kde=False,bins=30)


#we use spacy as our NLP tool  
import spacy
nlp = spacy.load('en_core_web_lg')
#resetin index is essential after droping values
train.reset_index(drop = True,inplace=True)

#train.drop(['index'],axis=1,inplace=True)
#dimiourgia df gia ta POS
train_POS = pd.DataFrame(data =np.zeros((train.shape[0],16), dtype=int),
                          columns = ['ADJ','ADP','ADV','CCONJ','DET','INTJ','NOUN','NUM','PART','PRON',
                                     'PROPN','PUNCT','SYM','VERB','X','SPACE'] )
result = pd.concat([train,train_POS],axis=1)
#gemisma tou pinaka me ta pos
for i in range(0,len(result)):
    print('We are dealing with word:',i)
    
    string = nlp(result['tweets'][i])
    POS_counts = string.count_by(spacy.attrs.POS)
    for k,v in sorted(POS_counts.items()):
        #print(f'{k}. {string.vocab[k].text:{5}}: {v}')
        result.at[i,string.vocab[k].text]=v


result.to_csv('got_users_with_POS.csv',index=False)