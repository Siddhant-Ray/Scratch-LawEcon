import json, pickle, argparse
import random
import math
from re import L

import torch 

from sentence_transformers import SentenceTransformer

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import numpy as np
from numpy import linalg as LA

from itertools import combinations

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

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
    # norm = (m * m).sum(0, keepdims=True) ** .5
    norm = LA.norm(m, axis = 1, keepdims = True)
    m_norm = m/norm; 
    similarity_matrix = m_norm @ m_norm.T 

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

# FILTER for different thresholds, find mean, save mean cosine values
def filter_matrixes_by_threshold_get_mean(cos_matrix, threshold):

    thr= float(threshold)

    mean_values = []
    thresholds = []
    num_elem = []

    for value in np.arange(thr, 0.15, -0.05):

        thresholds.append(value)
        # SET threshold for pairwise similarity
        masked_matrix = np.where(cos_matrix > value , 1, 0)
        # print(masked_matrix[0][0:15])
        indices_for_similar = np.where(masked_matrix==1)

        # Print first indices for entries above threshold
        # print(indices_for_similar[0][0:10])
        # print(indices_for_similar[1][0:10])

        values_above_threshold = cos_matrix[indices_for_similar[0], indices_for_similar[1]]
        print(values_above_threshold[0:10])
        mean_values.append(np.mean(values_above_threshold))
        num_elem.append(values_above_threshold.size)

    return thresholds, mean_values, num_elem

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-dev", "--device", help="device specifier")
    parser.add_argument("-sv", "--save", help = "choose saved numpy cosine matrix")
    parser.add_argument("-thr", "--threshold", help = "theshold for filtering cosine sim")

    args = parser.parse_args()

    print(args)

    if args.save:
        
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
            # print(list_of_embeddings.shape)

            pair_cosine_matrix = pairwise_cosine_sim_matrix(list_of_embeddings)
            # print(pair_cosine_matrix.shape)
            # print(pair_cosine_matrix[0][0:15])

            # np.savetxt("paraphrase/figs/cosine_sim.csv", pair_cosine_matrix, delimiter=",")
            np.save("paraphrase/data/cosine_sim.npy", pair_cosine_matrix)
            np.save("paraphrase/data/cosine_sim_16.npy", pair_cosine_matrix.astype(np.float16))
   
    else:
        print("Loading from saved.....")
        loaded_pair_cosine_matrix = np.load("paraphrase/data/cosine_sim_16.npy")
        print(loaded_pair_cosine_matrix.shape)
        print(loaded_pair_cosine_matrix[0][0:15])

        thresholds, mean_values, num_elem = filter_matrixes_by_threshold_get_mean(loaded_pair_cosine_matrix, args.threshold)
        plt.figure(1)
        plt.plot(thresholds, mean_values)
        plt.title("Threshold vs mean cosine similarity on satisfying indices")
        plt.xlabel("Threshold")
        plt.ylabel("Mean cosine similarity")
        plt.savefig("paraphrase/figs/threshold_cosine.png",format="png")

        plt.figure(2)
        plt.plot(thresholds, num_elem)
        plt.title("Threshold vs number of sentences above threshold")
        plt.xlabel("Threshold")
        plt.ylabel("Number of sentences above threshold")
        plt.savefig("paraphrase/figs/threshold_num.png",format="png")



if __name__ == '__main__':
    main()
        





