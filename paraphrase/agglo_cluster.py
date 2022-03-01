import json, pickle
from operator import le
import os, sys, time, math, random
import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer


PATH = "paraphrase/data/"
embedding_file_bbc = "paraphrase/data/test_corpus_bbc.pkl"

# LOAD numpy indices for pairs above a threshold for bbc
# We want equality pairs for the distance matrix, else 0 for (i,i)
# may give us wrong results
def load_indices_bbc(path):
    sent1_path= path + "sent1_indices_bbc.npy"
    sent2_path= path + "sent2_indices_bbc.npy"

    sent1_indices = np.load(sent1_path)
    sent2_indices = np.load(sent2_path)

    return sent1_indices, sent2_indices

# LOAD unique numpy indices for pairs above a threshold for bbc
# No (i,i) case, defined just in case
def load_unique_indices_bbc(path):
    sent1_path= path + "sent1_indices_noequal_bbc.npy"
    sent2_path= path + "sent2_indices_noequal_bbc.npy"

    sent1_indices = np.load(sent1_path)
    sent2_indices = np.load(sent2_path)

    return sent1_indices, sent2_indices

# Return paraphrase probabilities, unique or not unique 
def load_paraphrase_probs(file):
    probs = np.load(file)
    return probs

# LOAD embeddings from stored state
def load_embeddings(fname):
    with open(fname, "rb") as em:
        stored_data = pickle.load(em)
    
    return stored_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", "--data", help = "choose to take the bbc corpus")
    parser.add_argument("-noeq", "--noequal", help= "choose whether to include same sentences as pairs")
    

    args = parser.parse_args()

    if args.data == "bbc":
        if args.noequal:
            sent1_indices, sent2_indices = load_unique_indices_bbc(PATH)
            file = PATH + "para_probs_noequal_{}.npy".format(args.data)
            para_probs = load_paraphrase_probs(file)
            print("Loaded BBC pairs indices from saved, {} ones .....".format(args.noequal))
        else:
            sent1_indices, sent2_indices = load_indices_bbc(PATH)
            file = PATH + "para_probs_{}.npy".format(args.data)
            para_probs = load_paraphrase_probs(file)
            print("Loaded BBC pairs indices from saved with (i,i) pairs .....")

        print("indices shape", sent1_indices.shape, sent2_indices.shape)
        print("para_probs shape", para_probs.shape)
      
    else:
        pass 

    tuples_of_indices = zip(sent1_indices.tolist(), sent2_indices.tolist())
    matrix_init = np.zeros((para_probs.shape[0], para_probs.shape[0]))
    print("Empty matrix.......")
    print(matrix_init[0], matrix_init.shape)

    for ix, index in enumerate(tuples_of_indices):
        matrix_init[index] = para_probs[ix]

    print("Empty matrix filled with paraprobs.......")
    print(matrix_init[0], matrix_init.shape)

    sentence_embeddings = load_embeddings(embedding_file_bbc)['embeddings']
    print("Sentence vectors loaded from {} .....".format(args.data))
    print("Shape of sentence vectors", sentence_embeddings.shape)
    
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(4,12))  

    # Fit the sentence data to the visualizer
    visualizer.fit(sentence_embeddings)      
    # Finalize and render the figure 
    visualizer.show(outpath="paraphrase/figs/kelbow_kmeans_allsentences.png") 
    # Fit the paraprobs data to the visualizer

    _, ax = plt.subplots() # Create a new figure
    new_visualizer = KElbowVisualizer(model, k=(4,12), ax = ax)  
    new_visualizer.fit(para_probs.reshape(-1, 1))      
    # Finalize and render the figure 
    new_visualizer.show(outpath="paraphrase/figs/kelbow_kmeans_paraprobs.png") 


if __name__== '__main__':
    main()
