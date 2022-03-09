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
from sklearn.cluster import DBSCAN


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

# COMPUTE spectral clustering 
def custom_spectral_clustering(input_data, n_clusters, affinity):
    model = SpectralClustering(n_clusters=n_clusters, affinity = affinity,
                                assign_labels='discretize', random_state=42, n_jobs = -1)
    clusters = model.fit(input_data)
    labels = clusters.labels_
    return clusters, labels

#  COMPUTE DBSCAN clustering 
def custom_dbscan_clustering(input_data, metric):
    model = DBSCAN(metric = metric, n_jobs = -1)
    clusters = model.fit(input_data)
    labels = clusters.labels_
    return clusters, labels

# LOAD embeddings and sentences from stored state
def load_embeddings(fname):
    with open(fname, "rb") as em:
        stored_data = pickle.load(em)
    return stored_data


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
    parser.add_argument("-met", "--metric", help = "decide metric for DBSCAN clustering")
    parser.add_argument("-mtype", "--matrix_type", help="specify distance or similarity matrix")
    parser.add_argument("-nclus", "--nclusters", help="specify the number of clusters for agglomerative")
    parser.add_argument("-load", "--load", help="create csv from saved cluster labels")



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

    sentence_embeddings = load_embeddings(embedding_file_bbc)['embeddings']
    print("Sentence vectors loaded from {} .....".format(args.data))
    print("Shape of sentence vectors", sentence_embeddings.shape)

    tuples_of_indices = zip(sent1_indices.tolist(), sent2_indices.tolist())
    ### Shape of matrix should be sentence X sentences
    ### This matrix has the paraphrase probabilities for certain indices
    ### which indices have a paraphrase probability , with cosine sim 
    ### > 0.5 between the pair

    matrix_init = np.zeros((sentence_embeddings.shape[0], sentence_embeddings.shape[0]))
    print("Empty matrix.......")
    print(matrix_init[0], matrix_init.shape)

    for ix, index in enumerate(tuples_of_indices):
        matrix_init[index] = para_probs[ix]

    print("Similarity matrix filled with paraprobs.......")
    print(matrix_init[0], matrix_init.shape)

    ones_matrix = np.ones((sentence_embeddings.shape[0], sentence_embeddings.shape[0]))
    print("Empty ones matrix.......")
    print(ones_matrix[0], ones_matrix.shape)

    dist_matrix = ones_matrix - matrix_init
    print("Distance matrix.......")
    print(dist_matrix[0], dist_matrix.shape)

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
    n_clusters = int(args.nclusters) 

    if args.matrix_type == "sim":
        input_distance_matrix = matrix_init
    elif args.matrix_type == "dist":
        input_distance_matrix = dist_matrix
    else:
        print("input matrix type not selected")
        exit()

    if not args.load:
        print("generating clusters....")

        if args.model == "agglo":

            print("Linkage method used is {}".format(args.linkage))
            clustered_model, labels = custom_agglomerative_clustering(input_distance_matrix, n_clusters, args.linkage)
            print("Labels generated for agglomerative ......")
            print(labels.shape)
            print(labels[0:10])
            print("set of labels.....")
            print(set(labels.tolist()))

            np.save("paraphrase/data/agglo_labels_{}_{}_mtype_{}_nclusters_{}.npy".format(args.data, args.linkage,
                                                                                    args.matrix_type, str(n_clusters)), labels)

        elif args.model == "spectral":

            print("Affinity method used is {}".format(args.affinity))
            clustered_model, labels = custom_spectral_clustering(input_distance_matrix, n_clusters, args.affinity)
            print("Labels generated for spectral ......")
            print(labels.shape)
            print(labels[0:10])
            print("set of labels.....")
            print(set(labels.tolist()))

            np.save("paraphrase/data/spectral_labels_{}_mtype_{}.npy".format(args.data, args.matrix_type), labels)

        elif args.model == "dbscan":

            print("Metric used is {}".format(args.affinity))
            clustered_model, labels = custom_dbscan_clustering(input_distance_matrix, args.metric)
            print("Labels generated for dbscan ......")
            print(labels.shape)
            print(labels[0:10])
            print("set of labels.....")
            print(set(labels.tolist()))

            np.save("paraphrase/data/dbscan_labels_{}_mytpe_{}.npy".format(args.data, args.matrix_type), labels)

        else:
            print("specify model....")
            exit()

    elif args.load:
        print("loading clusters....")
        sentences = load_embeddings(embedding_file_bbc)["sentences"]
        print(sentences.shape)
        print(sentences[0:5])

        df = pd.DataFrame(sentences)

        numbers = [16, 32, 64, 128, 256, 512, 1024]

        for num in numbers:
        
            df["{} clusters".format(num)] = np.load("paraphrase/data/agglo_labels_{}_{}_mtype_{}_nclusters_{}.npy".format(args.data,
                                                                                            args.linkage, args.matrix_type, str(num)))
        # print(df.head())
        df.to_csv("paraphrase/figs/agglo_average_linkage_clustered.csv",index=False)

if __name__== '__main__':
    main()
