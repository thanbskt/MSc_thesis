# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 03:29:25 2020

@author: thanb
"""

#importing usefull libraries libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

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
                     'user_verified',
                     "user_screen_name"]]
#checking for miisng values
sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#droping rows without missing values or selecting the non-missing ones
dataset = dataset[dataset['user_id'].notna()]
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

#we drop columns we dont need anymore
dataset = dataset[["user_id",
                 'tweet_full_text',
                 "tweet_id",
                 'tweet_in_reply_to_user_id',
                 "tweet_retweet_count",
                 "user_favourites_count",
                 "user_followers_count",
                 "user_friends_count",
                 "user_profile_background_color",
                 "user_verified",
                 "user_screen_name"]]
# we drop rows where follows and friends count is zero
# these values we ll be in the denominator in the following formulas and we dont want devision with zero
dataset = dataset[dataset['user_followers_count'].ne(0)]
dataset = dataset[dataset['user_friends_count'].ne(0)]
dataset = dataset.reset_index(drop=True)


#we get the final list of users ids
users_ids = dataset['user_id'].unique().tolist()
users = pd.DataFrame(data = users_ids,columns = ['user_id']).sort_values(by=['user_id'])
mean_values = dataset.groupby('user_id').mean().sort_index().reset_index()
#dataset['tweet_user']= dataset['tweet_user'].apply(literal_eval)


#We try to implement the equations in the paper: MODELING TOPIC SPECIFIC CREDIBILITY ON TWITTER
users["RT_u"] = mean_values['tweet_retweet_count'].values
users["Fo_u"] = mean_values['user_followers_count'].values
users["Fe_u"] =  mean_values['user_friends_count'].values
users['tweets_per_user_tu'] = dataset['user_id'].value_counts().sort_index().values
#we get the verified value by dropiing duplicate rows of same user_id because
#a user may have different occurences in dataset but one verified value
users["verified"] =  dataset.drop_duplicates(subset=['user_id']).reset_index(drop=True)["user_verified"]

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

##In the following code we try to measure the formula with 2 different strategies
##First we measure all the variables of the final Cu formula and then we calculate the value.
## Final we logg and normalize the values between 1 to 0
##Second strategy is to measure the variables of final formula but we add 1(so we dont have negative values if we have values between 0,1)
## and we log in every variable. After that we calculate the final fomrula based on the loged values
## then we apply normalization. The difference is where logging takes place
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




#we visualize the distribution of verified users based on credibility score
users.sort_values(by="Cu_with_log_values")["verified"].index.values
data = np.asarray(users.sort_values(by="Cu_with_log_values")["verified"],dtype='float64')
sns.heatmap(data[:, np.newaxis], cmap='viridis')

sns.heatmap(users.sort_values(by="Cu_with_log_values")['verified'],cmap='viridis',linecolor = "black")

users["log_CredRT(u,x)"] = np.log(abs(users["RT_u"] - RT_mean )+1)
users["log_Utility(u,x)"] = np.log(abs(((users["RT_u"]*users["Fo_u"])/users['tweets_per_user_tu']) - ((RT_mean*Fo_mean)/tx) ) +1)
users["log_Credsocial(u)"] = np.log(abs((users["Fo_u"] /users['tweets_per_user_tu']) - (Fo_mean/t)) +1)
users["log_Balance_social(u)"] = np.log(abs((users["Fo_u"]/users["Fe_u"]) - (Fo_mean/Fe_mean)) +1)
#because we are dealing with one topic 
#equation (3) bocomes the same with equation (5)
users["log_Credsocial(u,x)"] = np.log(abs(users["Fo_u"]/users['tweets_per_user_tu'] - Fo_mean/tx)+1)
# and equation 6 becomes 1
users["log_Focus(u,x)"] = np.log(abs(users['tweets_per_user_tu'].sum()/users['tweets_per_user_tu'].sum()))
users["Cu_with_log_values"] = a*(users["log_Focus(u,x)"] + b*(users["log_Balance_social(u)"] * users["log_Credsocial(u)"])) + c*(users["Focus(u,x)"]*users["log_CredRT(u,x)"])

#we visualize the verified users again based on credibility score, we see no big diferrence
data = np.asarray(users.sort_values(by="Cu")["verified"],dtype='float64')
sns.heatmap(data[:, np.newaxis], cmap='viridis')


users["Cu_final_logged"] = np.log(users["Cu"])
test = pd.DataFrame(users[["user_id","Cu",'verified','Cu_with_log_values']])
test['loged_Cu'] = np.log(users["Cu"])

scaler = MinMaxScaler()

test["scaled_and_logged_Cu"]= scaler.fit_transform(test["loged_Cu"].values.reshape(-1,1))
test["logged_in_formula"] = users['Cu_with_log_values']
test [ "scaled_and_logged_in_formula"] = scaler.fit_transform(users["Cu_with_log_values"].values.reshape(-1,1))
#plotting the result from the two methods, logging in the formula or logging at the end,then we  scale both results
sns.distplot(test['scaled_and_logged_Cu'],kde=False,bins=50)
sns.distplot(test['scaled_and_logged_in_formula'],kde=False,bins=50)
plt.ylabel('count_of_users')
plt.xlabel("Credibility_of_user")
plt.show()

test["apply_verified_bonus"] = test["scaled_and_logged_Cu"]

#the verified users get s strong bonus in credibility score
#just a small portion of our users are verified
for i in range(0,len(test)):
    if test["verified"][i]:
        test["apply_verified_bonus"][i] = (test["scaled_and_logged_Cu"].max() + test["scaled_and_logged_Cu"][i])/2
    

#visualizing the verified users sorted by the Credibility score
#we can see now how the value bonus can affect the plot      
data = np.asarray(test.sort_values(by="apply_verified_bonus")["verified"],dtype='float64')
sns.heatmap(data[:, np.newaxis], cmap='viridis')


#we can figure out that logging and then normalizing is better to keep better value range
#values tend to be less sesitive and tend to be less extreme and thus more interpretable

def edit(df):
    my_df = df.copy()
    my_df["user_id"] = my_df["user_id"] + 10
    return my_df

new_df = edit(dataset)





#we create a new column in users dataset and we create a corpus of tweets
#to create the corpus we concated every tweet per user
users["tweets"] = ""

for k,v in enumerate(users_ids):
    print(k,v)
    m = dataset[dataset["user_id"] == v].reset_index(drop=True)  
    for i in range(0,len(m)):
        print(i)
        users["tweets"][k] =  users["tweets"][k] + " " +  m["tweet_full_text"][i]
        
