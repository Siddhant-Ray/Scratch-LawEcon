import json, pickle
import random
import math
import torch 

from sentence_transformers import SentenceTransformer

import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data_file = pd.read_csv("paraphrase/test_corpora/source_corpus2.csv")
model = SentenceTransformer('all-MiniLM-L6-v2', device = device)

print(data_file.head())

list_of_sentences = data_file['text'].to_list()
list_of_embeddings = model.encode(list_of_sentences)

with open('paraphrase/data/test_corpus1.pkl', "wb") as fOut1:
    pickle.dump({'sentences': list_of_sentences, 'embeddings': list_of_embeddings}, fOut1, protocol=pickle.HIGHEST_PROTOCOL)
