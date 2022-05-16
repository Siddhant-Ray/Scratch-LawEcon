import pandas as pd
import os, sys
import relatio
import numpy as np
import pickle

from relatio.wrappers import run_srl
from relatio.utils import split_into_sentences
from relatio.wrappers import build_narrative_model

import nltk
nltk.download('averaged_perceptron_tagger')

import spacy
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

## Flags 

split_sents = False
run_srl_method = False
run_narrative = True

data_path = "labour_contracts/data/relatio_formatted.csv"

## load data 
data_frame = pd.read_csv(data_path)
print(data_frame.head(10))
print(data_frame.shape)

new_cols = ["id", "doc"]
new_frame = data_frame[["id", "text"]]
new_frame.columns = new_cols
new_frame.fillna('', inplace=True)

print(new_frame.head(10), new_frame.shape)

new_short_frame = new_frame.head(500)

if split_sents:

    split_sentences = split_into_sentences(new_short_frame, progress_bar=True)

    with open('labour_contracts/data/sentences.pkl', "wb") as sent_file:
        pickle.dump({'id': split_sentences[0], 'sentences': split_sentences[0]}, sent_file, protocol=pickle.HIGHEST_PROTOCOL)

cuda_str = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
cuda_device = int(cuda_str[0]) if cuda_str[0] else -1
print(f"Using CUDA:{cuda_device}")

# USE GPU
if run_srl_method:

    with open("labour_contracts/data/sentences.pkl", 'rb') as f:
        sentences_file = pickle.load(f)

    ids, sentences = sentences_file["id"], sentences_file["sentences"]
    print(len(ids), len(sentences))

    srl_res = run_srl(
                path = "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz", # pre-trained model
                sentences = sentences,
                cuda_device = cuda_device,
                progress_bar = True)

    print(len(srl_res))

    with open('labour_contracts/data/srl.pkl', "wb") as sent_file:
        pickle.dump({'srl' : srl_res , 'sentences': sentences}, sent_file, protocol=pickle.HIGHEST_PROTOCOL)

if run_narrative:

    with open('labour_contracts/data/srl.pkl', 'rb') as f:
        file_load = pickle.load(f)

    srl_res = file_load['srl']
    sentences = file_load['sentences']

    print(srl_res[0:10])
    print(sentences[0:10])

    print(len(sentences), len(srl_res))

    narrative_model = build_narrative_model(
        srl_res=srl_res,
        sentences=sentences,
        embeddings_type="gensim_keyed_vectors", 
        embeddings_path="glove-wiki-gigaword-100",
        n_clusters=[[100]],
        top_n_entities=100,
        stop_words = spacy_stopwords,
        remove_n_letter_words = 1,
        progress_bar=True,
    )

    print(narrative_model['entities'].most_common()[:20])

    with open('labour_contracts/data/model.pkl', "wb") as sent_file:
        pickle.dump({'model' : narrative_model}, sent_file, protocol=pickle.HIGHEST_PROTOCOL)


