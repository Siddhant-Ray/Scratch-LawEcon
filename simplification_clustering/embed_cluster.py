import os, json, pickle, sys, time
import argparse
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans

PATH = "simplificaiton_clustering/datasets/ABCD/"

# Get embeddings
def embedd_sentences(sentences):
    sbert_model = 'all-MiniLM-L6-v2'
    max_seq_length = 128
    embedder = SentenceTransformer(sbert_model)
    embedder.max_seq_length = max_seq_length
    corpus_embeddings = embedder.encode(sentences, batch_size=32, device = 'cuda', show_progress_bar=True)
    return corpus_embeddings

# Get clusters
def run_kmeans(X, n_clusters=44):
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=512).fit(X)
        labels = kmeans.labels_
        return labels

# Load data
def load_data(path):
    data = open(path+"bbc_data_splitted.txt", 'r').readlines()
    print(data)

# Run embeddings and clustering
def run(args):
    load_data(args.path)

# Main
def main():
    parser = argparse.ArgumentParser(description='Run K-Means clustering on the dataset')
    parser.add_argument('--path', type=str, default=PATH, help='Path to the dataset')
    parser.add_argument('--n_clusters', type=int, default=44, help='Number of clusters')
    args = parser.parse_args()

    run(args)

if __name__ == '__main__':
    main()
    