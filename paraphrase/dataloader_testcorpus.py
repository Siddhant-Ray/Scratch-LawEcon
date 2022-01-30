import json, pickle, argparse
import random
import math
import torch 

from sentence_transformers import SentenceTransformer

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import numpy as np

from itertools import combinations

import pandas as pd

file = "paraphrase/test_corpora/source_corpus2.csv"
stored_file = "paraphrase/data/test_corpus1.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('all-MiniLM-L6-v2', device = device)
print(device, model)

# GET corpus as list of sentences
def get_corpus(full_file_path):

    data_file = pd.read_csv(full_file_path)
    list_of_sentences = data_file['text'].to_list()

    return list_of_sentences

# SLOW pair calculation from list of sentences
def get_pairs_of_sentences(sentences):
    
    sentence_pairs = list(combinations(sentences,2))

    sentence1 = sentence_pairs[:, 0]
    sentence2 = sentence_pairs[:, 1]

    return sentence1, sentence2

# FAST cosine pairwise function
def pairwise_cosine_sim_matrix(input_matrix):
    m = input_matrix
    norm = (m * m).sum(0, keepdims=True) ** .5
    m_norm = m/norm; 
    similarity_matrix = m_norm.T @ m_norm

    return similarity_matrix

# SAVE embeddings for next time
def generate_and_save_embeddings(sentences):
    
    list_of_sentences = sentences
    list_of_embeddings = model.encode(sentences)
    
    with open('paraphrase/data/test_corpus1.pkl', "wb") as fOut1:
        pickle.dump({'sentences': list_of_sentences, 'embeddings': list_of_embeddings}, fOut1, protocol=pickle.HIGHEST_PROTOCOL)

    return list_of_embeddings, list_of_sentences

# LOAD embeddings from stored state
def load_embeddings(fname):

    with open(fname, "rb") as em:
        stored_data = pickle.load(em)
    
    return stored_data


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-dev", "--device", help="device specifier")

    args = parser.parse_args()

    print(args)

    if args.device == "gpu":
        print("generating new embeddings........")
        sentences = get_corpus(file)
        print(sentences[0:5])
        list_of_embeddings, list_of_sentences = generate_and_save_embeddings(sentences)
        print(list_of_embeddings.shape)

    elif args.device == "cpu":
        print("loading stored embeddings........")
        stored_data = load_embeddings(stored_file)
        list_of_embeddings = stored_data['embeddings']
        print(list_of_embeddings.shape)

if __name__ == '__main__':
    main()
        





