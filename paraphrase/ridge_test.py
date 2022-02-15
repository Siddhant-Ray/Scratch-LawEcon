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
model_path = "paraphrase/saved_models/_ridge_full.sav"

#FILTER corpus based on indices
def filter_corpus_as_dataframe(full_file_path, list_of_indices):
    data_file = pd.read_csv(full_file_path)['text']
    df_new = data_file.iloc[list_of_indices]
    return df_new

# LOAD embeddings from stored state
def load_embeddings(fname):

    with open(fname, "rb") as em:
        stored_data = pickle.load(em)
    
    return stored_data

# FAST cosine pairwise function
def weighted_pairwise_cosine_sim_matrix(input_matrix1, input_matrix2):
    m1 = input_matrix1
    m2 = input_matrix2
    # norm = (m * m).sum(0, keepdims=True) ** .5
    norm1 = LA.norm(m1, axis = 1, keepdims = True)
    norm2 = LA.norm(m2, axis = 1, keepdims = True)
    m1_norm = m1/norm1
    m2_norm = m2/norm2
    similarity_matrix = m1_norm @ m2_norm.T 

    return similarity_matrix

# LOAD the weights of the trained ridge model
def load_saved_model_coeff(model_path):
    loaded_model = pickle.load(open(model_path, 'rb'))
    return loaded_model, loaded_model.coef_

# Pre multiply each vector once all by of the model coeffs, so that in the dot product
# y_hat = dot(u,v,w), it appears once
def pre_multiply_vectors_by_model_weights(input_vectors, model_coeff):
    scaled_output_vectors = input_vectors * model_coeff
    return scaled_output_vectors

# For one threshold, get all incides for satisfying pairs
def filter_for_single_threshold(cos_matrix, threshold):
    thr= float(threshold)

    masked_matrix = np.where(cos_matrix > thr , 1, 0)
    indices_for_similar = np.where(masked_matrix==1)
    
    return indices_for_similar[0], indices_for_similar[1]

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-load", "--load_model", help = "Specify if a model is to be loaded")
    parser.add_argument("-eload", "--load_embeddings", help = "Specify embeddings file to be loaded")
    parser.add_argument("-th", "--threshold", help = "Specify threshold for filtering")
    args = parser.parse_args()

    if args.load_model:
        model, model_coeff = load_saved_model_coeff(model_path)
        print("Model coeff shape", model_coeff.shape)

    if args.load_embeddings:
        stored_data = load_embeddings(stored_file)
        sent_vectors = stored_data['embeddings']
        print("Input sent vectors shape", sent_vectors.shape)

    weighted_vectors = pre_multiply_vectors_by_model_weights(sent_vectors, model_coeff)
    print("Weighted_vectors_shape", weighted_vectors.shape)
    #print(weighted_vectors[0:10])

    weighted_cosine_similarity = weighted_pairwise_cosine_sim_matrix(weighted_vectors, sent_vectors)
    print(weighted_cosine_similarity.shape)
    print(weighted_cosine_similarity[0:10])

    indices_for_sent1, indices_for_sent2 =  filter_for_single_threshold(weighted_cosine_similarity, args.threshold)
    print("No of sentence pairs above threshold")
    print(indices_for_sent1.shape)
    print(indices_for_sent2.shape)

    df_new1 = filter_corpus_as_dataframe(file, indices_for_sent1.tolist())
    df_new2 = filter_corpus_as_dataframe(file, indices_for_sent2.tolist())

    print(df_new1.shape, df_new2.shape)
    print(df_new1.head())
    print(df_new2.head())



if __name__ == '__main__':
    main()