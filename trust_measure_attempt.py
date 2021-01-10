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

dataset = dataset[[  'user_id',
                     'tweet_created_at',
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
                     'user_profile_background_color',
                     'user_verified']]
#checking for miisng values
sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#droping rows without missing values or selecting the non-missing ones
dataset = dataset[dataset['user_id'].notna() ]
#sorting our values based on user_id
dataset = dataset.sort_values(by=['user_id']).reset_index(drop=True)

#visualize the most common colors of backgraound profile
colors = dataset['user_profile_background_color'].value_counts().index.values.tolist()
color_counts = dataset['user_profile_background_color'].value_counts().values.tolist()
colors = colors[:20]
color_counts=color_counts[:20]
        #
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
plt.show
#visualize the retweets_count
x= dataset['tweet_retweet_count'].index.values
y=dataset['tweet_retweet_count'].values
plt.ylabel('number_of_retweets')
plt.xlabel('tweet_id')
plt.scatter(x,y)
plt.show()

dataset = dataset[["user_id",
                 'tweet_full_text',
                 "tweet_id",
                 'tweet_in_reply_to_user_id',
                 "tweet_retweet_count",
                 "user_favourites_count",
                 "user_followers_count",
                 "user_friends_count",
                 "user_profile_background_color",
                 "user_verified"]]







#dataset['user_id'] = dataset['user_id'].astype('Int64')

#we isolate the unique user by their id
users_ids = dataset['user_id'].unique().tolist()
users = pd.DataFrame(data = users_ids,columns = ['user_id']).sort_values(by=['user_id'])
mean_values = dataset.groupby('user_id').mean().sort_index()
#dataset['tweet_user']= dataset['tweet_user'].apply(literal_eval)



#We try to implement the equations in the paper: MODELING TOPIC SPECIFIC CREDIBILITY ON TWITTER
users["RT_u"] = mean_values['tweet_retweet_count'].values
users["Fo_u"] = mean_values['user_followers_count'].values
users["Fe_u"] =  mean_values['user_friends_count'].values
users['tweets_per_user_tu'] = dataset['user_id'].value_counts().sort_index().values
user['verified'] = dataset['user_verified']

RT_mean = users["RT_u"].mean()
Fo_mean = users['Fo_u'].mean()
Fe_mean = users['Fe_u'].mean()
tx = t = len(dataset)
#equation (1) CredRT(u,x) = |RTu - RTx_mean|

#equation (2) Utility(u,x) = |RTu,x*Fo(u)/t(u,x) -RTx_mean * F(o,x)_mean/tx 

#equation (3) Credsocial(u) = |Fo(u)/tu -Fo_mean/t_total | 

#equation (4) Balance_social(u) = |Fo(u)/Fe(u)  -  Fo_mean/Fe_mean|

#equation (5) Credsocial(u,x) = |Fo(u,x)/tu,x  - Fo,x_mean/tx|

#equation (6) Focus(u,x)      = |Sum of all tu,x/ sum of all tu|
#for just one topic we get:
#RTx_mean -> RT_mean
#RTu,x    -> RTu  
#Fo,x     -> Fo
#tu,x     ->tu (tweets of user u)
#tx       ->t  (tweets in total)

users["CredRT(u,x)"] = abs(users["RT_u"] - RT_mean )
users["Utility(u,x)"] = abs(((users["RT_u"]*users["Fo_u"])/users['tweets_per_user_tu']) - ((RT_mean*Fo_mean)/tx) )
users["Credsocial(u)"] = abs((users["Fo_u"] /users['tweets_per_user_tu']) - (Fo_mean/t))
users["Balance_social(u)"] = abs((users["Fo_u"]/users["Fe_u"]) - (Fo_mean/Fe_mean))
#because we are dealing with one topic 
#equation (3) bocomes the same with equation (5)
users["Credsocial(u,x)"] = abs(users["Fo_u"]/users['tweets_per_user_tu'] - Fo_mean/tx)
# and equation 6 becomes 1
users["Focus(u,x)"] = abs(users['tweets_per_user_tu'].sum()/users['tweets_per_user_tu'].sum())


a=0.5
b=0.3
c= 0.5

users["Cu"] = a*(users["Focus(u,x)"] + b*(users["Balance_social(u)"] * users["Credsocial(u)"])) + c*(users["Utility(u,x)"]*users["CredRT(u,x)"])
users[]
test = dataset[dataset['user_id'] ==  3307430803]

X = dataset [dataset['user_id'] ==  users[1]]
testttt = dataset.set_index(['user_id'])

test2=dataset.stack()
dataset.groupby('user_id').cumcount()


testttt.sort_index(inplace=True)

dataset.nunique()

























