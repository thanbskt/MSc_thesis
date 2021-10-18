# MSc_thesis
repository for my master thesis in NLP with deep learning

The dataset can be found in this kaggle link: https://www.kaggle.com/monogenea/game-of-thrones-twitter

The purpose of the master thesis is to measure the user credibility and predict it based on text data we preprocessed using Natural Language Preprocessing. The master thesis consists of 3 steps.

1) In the first step we explore our data and measure the user credibility based on a paper, the code for this process can be found on the measure_trust.py file.


2) In the second step we preprocess our data and we create new NLP features like POS tags. We concat them on the initial dataset and we group our data by the users. We add the text data for every user according to their messages.  
The code for this step is on files got_preprocessing_1.py and got_preprocessing_2.py

3)In the final step we produce the hybrid deep learning model that can be train both on numerical and text data using word embeddings technique. In the end we evaluate our data using accuracy and validation accuracy in file hybrid_model.py