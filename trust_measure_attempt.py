# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 03:29:25 2020

@author: thanb
"""

#importing usefull libraries libraries
import pandas as pd
import numpy as np
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns

#adjusting display settings for better visualization
dataset = pd.read_csv('justdoit_tweets_2018_09_07_2.csv',  parse_dates=['tweet_created_at', 'user_created_at'])
pd.set_option('display.max_colwidth', 10000)
pd.set_option('display.max_columns', None)

dataset = dataset[[  'tweet_created_at',
                     'tweet_entities',
                     'tweet_full_text',
                     'tweet_id',
                     'tweet_in_reply_to_user_id',
                     'tweet_lang',
                     'tweet_retweet_count',
                     'tweet_retweeted',
                     'tweet_user',
                     'user_favourites_count',
                     'user_followers_count',
                     'user_following',
                     'user_friends_count',
                     'user_id',
                     'user_profile_background_color',
                     'user_verified']]
#visualize the most common colors of backgraound profile
colors = dataset['user_profile_background_color'].value_counts().index.values.tolist()
color_counts = dataset['user_profile_background_color'].value_counts().values.tolist()
colors = colors[:20]
color_counts=color_counts[:20]

for count,values in enumerate(colors):
    colors[count] = "#"+values
    print(count,values,"  ",colors[count])

plt.bar(colors,color_counts ,color=colors)
plt.xticks(rotation=30)
plt.show()

#visualize tweets count over some time period
tweet_dataset_5min = dataset.groupby(pd.Grouper(key='tweet_created_at', freq='15Min', convention='start')).size()
tweet_dataset_5min.plot(figsize=(18,6))
plt.ylabel('5 Minute Tweet Count')
plt.title('Tweet Freq. Count')
plt.grid(True)
#visualize the retweets_count
x= dataset['tweet_retweet_count'].index.values
y=dataset['tweet_retweet_count'].values
plt.scatter(x,y)
plt.show()





dataset.nunique()

dataset['tweet_user']= dataset['tweet_user'].apply(literal_eval)
#afairoume tous xristes pou den exoun user_id
dataset['user_id'].dropna(inplace=True)

dataset['user_id'] = dataset['user_id'].astype('Int64')
#we isolate the unique user by their id
users = dataset['user_id'].unique().astype('Int64')

X = dataset [dataset['user_id'] ==  users[0]]

test = dataset[dataset['user_id'].isnull()]





type(Y['id'])
X
Y = X[1249]
keys = []
values = []
for k,d in X[1249]:
	print(k,d)
    
Y['id']
type(X)
y = dataset['user_id'].unique()
len(y)
    
type(dataset['tweet_created_at'][0])




line1 = ax1.plot(x, myvalues,'#333399', marker='o', label='My Blue values')

















