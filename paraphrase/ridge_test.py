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
model_path = "paraphrase/saved_models/ridge_full.sav"

# LOAD embeddings from stored state
def load_embeddings(fname):

    with open(fname, "rb") as em:
        stored_data = pickle.load(em)
    
    return stored_data

# FAST cosine pairwise function
def pairwise_cosine_sim_matrix(input_matrix):
    m = input_matrix
    # norm = (m * m).sum(0, keepdims=True) ** .5
    norm = LA.norm(m, axis = 1, keepdims = True)
    m_norm = m/norm; 
    similarity_matrix = m_norm @ m_norm.T 

    return similarity_matrix

# LOAD the weights of the trained logistic model
def load_saved_model_coeff(model_path):
    loaded_model = pickle.load(open(model_path, 'rb'))
    return loaded_model, loaded_model.coef_

def pre_multiply_vectors_by_model_weights(input_vectors, model_coeff):
    return None

def main():
     parser = argparse.ArgumentParser()
     parser.add_argument("-load", "--load_model", help = "Specify if a model is to be loaded")
     args = parser.parse_args()

     if args.load:
         model, model_coeff = load_saved_model_coeff(model_path)
         print(model_coeff.shape)




if __name__ == '__main__':
    main()