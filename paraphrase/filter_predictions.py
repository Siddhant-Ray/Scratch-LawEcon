import json, pickle
from operator import le
import os, sys, time, math, random
import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

import spacy   
from spacy.matcher import Matcher
from spacy.util import filter_spans
nlp = spacy.load('en_core_web_sm')

import string

# Get all verbs using a Spacy based function
def get_verbs(input_sentence):

    sentence = input_sentence
    pattern = [{'POS': 'VERB', 'OP': '?'},
            {'POS': 'ADV', 'OP': '*'},
            {'POS': 'AUX', 'OP': '*'},
            {'POS': 'VERB', 'OP': '+'}]

    # instantiate a Matcher instance
    matcher = Matcher(nlp.vocab)
    matcher.add("Verb phrase", [pattern])

    doc = nlp(sentence) 
    # call the matcher to find matches 
    matches = matcher(doc)
    spans = [doc[start:end] for _, start, end in matches]

    return (filter_spans(spans))

# Get sentence length
def get_sentence_length(input_sentence):
    _list = word_tokenize(input_sentence)
    return len(_list)

# FILTER sentences in the DataFrame
def filter_sent_in_df(df):
    print(df.shape)
    print(df.head())
    for character in string.punctuation:
        # print(character)
        df = df[df.sent1.str.replace(character,"",regex=False) != df.sent2.str.replace(character,"",regex=False)]
    df = df[df.sent1.str.replace("—","",regex=False) != df.sent2.str.replace("—","",regex=False)]
    df = df[df.sent1.str.replace("“( )","",regex=False) != df.sent2.str.replace("“( )","",regex=False)]
    df = df.drop(df[df.sent1.str.contains("( )", regex = False)].index)
    df = df[df.sent1.str.len() >= 50]
    df = df[df.sent2.str.len() >= 50]

    df = df[df.sent1.str.len() <= 300]
    df = df[df.sent1.str.len() <= 300]

    return df



def main():
    dataframe = pd.read_csv("paraphrase/figs/top_100000_noequal_bbc.csv")

    df = filter_sent_in_df(dataframe)
    df0 = pd.DataFrame({"para_probs": df.para_probs})

    df1 = pd.DataFrame({"sent1": df.sent1})
    df1['sent1_verbs'] = df1['sent1'].apply(lambda row: get_verbs(row))
    df1['sent1_length'] = df1['sent1'].apply(lambda row: get_sentence_length(row))

    df2 = pd.DataFrame({"sent2": df.sent2})
    df2['sent2_verbs'] = df2['sent2'].apply(lambda row: get_verbs(row))
    df2['sent2_length'] = df2['sent2'].apply(lambda row: get_sentence_length(row))

    print("DF 1.......", df1.shape)
    print(df1.head())
    print("DF 2.......", df2.shape) 
    print(df2.head())

    df3 = (df1['sent1_verbs'].str.len() - df2['sent2_verbs'].str.len()).abs() 
    df4 = (df1['sent1_length'] - df2['sent2_length']).abs()

    df3 = pd.DataFrame({"num_verbs_diff": df3})
    df4 = pd.DataFrame({"sent_lent_diff": df4})

    df_final = pd.concat([df0, df1, df2, df3, df4], axis = 1)
    #df_final.to_csv("paraphrase/figs/filtered_bbc.csv",index=False)
    df_final.to_csv("paraphrase/figs/filtered_bbc_further.csv",index=False)

if __name__== '__main__':
    main()
