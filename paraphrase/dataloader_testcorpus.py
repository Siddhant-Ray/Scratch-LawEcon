import json, pickle
import random
import math
import torch 

from sentence_transformers import SentenceTransformer

from itertools import combinations

import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data_file = pd.read_csv("paraphrase/test_corpora/source_corpus2.csv")
model = SentenceTransformer('all-MiniLM-L6-v2', device = device)

print(data_file.head())

list_of_sentences = data_file['text'].to_list()

def get_pairs_of_sentences(sentences):
    
    sentence_pairs = list(combinations(sentences,2))

    sentence1 = sentence_pairs[:, 0]
    sentence2 = sentence_pairs[:, 1]

    return sentence1, sentence2

s1, s2 = get_pairs_of_sentences(list_of_sentences)

print(len(s1))
print(len(s2))

# list_of_embeddings1 = model.encode(s1)
# list_of_embeddings2 = model.encode(s2)

# print(list_of_embeddings1.shape)
# print(list_of_embeddings2.shape)

'''with open('paraphrase/data/test_corpus1.pkl', "wb") as fOut1:
    pickle.dump({'sentences': list_of_sentences, 'embeddings': list_of_embeddings}, fOut1, protocol=pickle.HIGHEST_PROTOCOL)'''
