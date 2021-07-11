# -*- coding: latin-1 -*-

""" APPLICATION DOCUMENTATION
    -------------------------
    Author:     Rebecca Wright
    Date:       July 11, 2021

    Purpose:    To identify underlying similarities between job postings.  Comparison excludes phrases including: job titles, locations, and equal-opportunity-employment disclaimer language.
                By isolating commonly occurring phrases and keywords, a user can more easily identify similarly appealing job listings in the future.  Results are isolated to top-5 single, double, and triple word phrases.
    
    Future Application:     Results will be harnessed to train recommendation engine for identification of personalized preferential job postings.


    Parameters: json file containing any number of job objects in the expected format of:
        [
            {
                "title": ...,
                "company_name": ...,
                "location": ...,
                "via": ...,
                "description": ...,
                "job_id": ...
            },
            {
                ...
            }
        ]

    Returns: Isolates top-5 single, double, and triple word phrases as a json file titled "nlp_results.json", written out to the same filepath location in the following format:
        {
            "num_jobs_compared":    // integer count of number of job objects within passed parameter
            "job_id_list":          // array of strings of actual job ids used during analysis
            "results_uni":          // array of 5 one-word strings identified as top occuring values during analysis
            "results_bi":           // array of 5 two-word strings identified as top occuring values during analysis
            "results_tri":          // array of 5 three-word strings identified as top occuring values during analysis
        }​
"""


# *****************************************************************************************************************************************************
# *****************************************************************************************************************************************************
# IMPORTS

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from wordcloud import STOPWORDS
from textblob import TextBlob
import string
import sys
import json

# *****************************************************************************************************************************************************
# PREPROCESSING FUNCTIONS

""" Function name: my_preprocessor
    Parameters: text
    Returns: text as lowercase, stripped of non-alpha characters and extra spaces"""
def my_preprocessor(text):
    text = [word for word in text.lower().split() if word not in string.punctuation]    #POSSIBLY REDUNDANT LINE
    text = [re.sub('[^a-zA-Z]','', c) for c in text]  # remove none alpha character
    text = [word for word in text if len(word)>0 if word not in custom_stopwords]     # remove empty character elements from list
    return ' '.join(text)

""" Function name: lemmatize_with_postag
    Passed: sentence
    Returns: lemmatized text, where the choices made be TextBlob lemmatizer are based on POS tag"""
def lemmatize_with_postag(sentence):
    sent = TextBlob(sentence)
    tag_dict = {"J": 'a',
                "N": 'n',
                "V": 'v',
                "R": 'r'}
    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]
    lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
    return lemmatized_list

# *****************************************************************************************************************************************************
# HELPER FUNCTIONS

""" Function name: top_words
    Parameters: X_vect and vectorizer
    Returns: top 5 occuring terms as dataframe"""
def top_words(X, vect):
    temp_df = pd.DataFrame(X.toarray(), columns = vect.get_feature_names())
    temp_df = temp_df.sum().sort_values(ascending=False).head(5)
    temp_df = temp_df.reset_index()
    return (temp_df['index'])

""" Function name: get_unique
    Parameters: column_name string and dataframe
    Returns: set of unique words found in column"""
def get_unique(column_name, df):
    words = df[column_name].str.lower().str.findall("\w+")
    unique_words = set()
    for x in words:
        unique_words.update(x)
    return unique_words

# *****************************************************************************************************************************************************
# STOPWORD COLLECTION FUNCTION

""" Function name: create_stopwords
    Parameters: DF name
    Returns: list collection of custom stopwords"""
def create_stopwords(df):
    custom_stopwords = list(STOPWORDS)    # initialize base stopwords from wordcloud
    words_with_punc = [word for word in STOPWORDS if not all(ch not in word for ch in string.punctuation)]    # isolate base stopwords that contain punctuation
    stripped_wwp = [re.sub('[^a-zA-Z]','', c) for c in words_with_punc]    # strip punctuation from words_with_punc for safety (covers stopword stripping both pre/post nlp preprocessing)
    # expand stopwords to include common keywords from equal opportunity employment disclaimer
    EOE_words = [
        'equal',
        'opportunity',
        'sexual',
        'orientation',
        'race',
        'gender',
        'martial',
        'status',
        'color',
        'religion',
        'disability',
        'veteran',
        'origin',
        'identity']
    [custom_stopwords.append(word) for word in stripped_wwp]    # add stripped_wwp to custom_stopwords
    [custom_stopwords.append(word) for word in get_unique("title", df)]    # add unique title words to custom_stopwords
    [custom_stopwords.append(word) for word in get_unique("location", df)]    # add unique location words to custom_stopwords
    [custom_stopwords.append(word) for word in get_unique("company_name", df)]    # add unique company words to custom_stopwords
    [custom_stopwords.append(word) for word in EOE_words]    # add EOE disclaimer words to custom_stopwords
    return custom_stopwords

# *****************************************************************************************************************************************************
# PRIMARY COMPARISON FUNCTION

""" Function name: run_comparison
    Parameters: two integers for ngram paramteter values
    Returns: lemmatized text, where the choices made be TextBlob lemmatizer are based on POS tag"""
def run_comparison (n1, n2):
    custom_tf = TfidfVectorizer(preprocessor=my_preprocessor, tokenizer=lemmatize_with_postag, max_features = 500, ngram_range = (n1, n2))
    data_custom_tf = custom_tf.fit_transform(data)
    custom_tf_df = pd.DataFrame(data_custom_tf.toarray(), columns = custom_tf.get_feature_names())
    return top_words(data_custom_tf,custom_tf)

# *****************************************************************************************************************************************************
# *****************************************************************************************************************************************************
# *****************************************************************************************************************************************************

# MAIN APPLICATION

filename = sys.argv[1]

df = pd.read_json(filename)

custom_stopwords = create_stopwords(df)

data = df['description']

results_uni = run_comparison(1,1)
results_bi = run_comparison(2,2)
results_tri = run_comparison(3,3)

results = {
    "num_jobs_compared": df.shape[0],
    "job_id_list": list(df['job_id']),
    "results_uni": list(results_uni),
    "results_bi": list(results_bi),
    "results_tri": list(results_tri)
}

with open("nlp_results.json", "w") as write_file:
    json.dump(results, write_file)
