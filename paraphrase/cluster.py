import json, pickle
from operator import le
import os, sys, time, math, random
import argparse
from tkinter import N

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering


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

# Function for kelbow plots generic
def kelbow_visualize(input_data, clf, title, out_path):
    _, ax = plt.subplots() # Create a new figure
    model = clf
    visualizer = KElbowVisualizer(model, k=(4,12), title = title, ax = ax) 
    visualizer.fit(input_data)
    visualizer.show(outpath = out_path) 

# COMPUTE agglomerative clustering 
def custom_agglomerative_clustering(input_data, n_clusters, linkage):
    model = AgglomerativeClustering(n_clusters, affinity = "precomputed", linkage=linkage)
    clusters = model.fit(input_data)
    labels = clusters.labels_
    return clusters, labels

# Compute spectral clustering 
def custom_spectral_clustering(input_data, n_clusters, affinity):
    model = SpectralClustering(n_clusters=n_clusters, affinity = affinity,
                                assign_labels='discretize', random_state=42, n_jobs = -1)
    clusters = model.fit(input_data)
    labels = clusters.labels_
    return clusters, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", "--data", help = "choose to take the bbc corpus")
    parser.add_argument("-noeq", "--noequal", help= "choose whether to include same sentences as pairs")
    parser.add_argument("-clf", "--classifier", help= "choose classifier for kelbow", default = KMeans())
    parser.add_argument("-mtx", "--matrix", help = "specify the matrix of input")
    parser.add_argument("-vis", "--visualize", help = "decide if kelbow should be plotted")
    parser.add_argument("-mod","--model", help = "clustering model selection", required = True)
    parser.add_argument("-link", "--linkage", help = "decide linkage for agglomerative clustering")
    parser.add_argument("-aff", "--affinity", help = "decide affinity for spectral clustering")

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

    if args.classifier == "kmeans":
         model = KMeans(random_state = 42)

    if args.matrix == "sentences":
        input_data = sentence_embeddings
        title = "Run on sentence embeddings"
        out_path = "paraphrase/figs/kelbow_kmeans_allsentences.png"
    elif args.matrix == "paraprobs":
        input_data = para_probs.reshape((-1,1))
        title = "Run on para probs"
        out_path = "paraphrase/figs/kelbow_kmeans_paraprobs.png"

    if args.visualize:
        kelbow_visualize(input_data, model, title, out_path)

    ## From the kelblow plots, we have k = 7 
    n_clusters = 7 
    input_distance_matrix = matrix_init

    if args.model == "agglo":

        print("Linkage method used is {}".format(args.linkage))
        labels, clustered_model = custom_agglomerative_clustering(input_distance_matrix, n_clusters, args.linkage)
        print("Labels generated for agglomerative ......")
        print(labels.shape)
        print(labels[0:10])

        np.save("paraphrase/data/agglo_labels_{}.npy".format(args.data), labels)

    if args.model == "spectral":

        print("Affinity method used is {}".format(args.affinity))
        labels, clustered_model = custom_spectral_clustering(input_distance_matrix, n_clusters, args.affinity)
        print("Labels generated for spectral ......")
        print(labels.shape)
        print(labels[0:10])

        np.save("paraphrase/data/spectral_labels_{}.npy".format(args.data), labels)

if __name__== '__main__':
    main()
