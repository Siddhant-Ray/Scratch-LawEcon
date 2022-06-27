import os, json, pickle, sys, time
import argparse
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans

PATH = "simplification_clustering/datasets/"

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
    non_simplified_sentences = []
    simplified_sentences = []

    data = open(path+"bbc_data_complex_splitted.txt", 'r').readlines()
    for line in data:
        line_val = json.loads(line)
        for item in line_val['simplified']:
            if len(item["text"].split()) >=6:   
                simplified_sentences.append(item['text'])

    sdata = open(path+"bbc_data_simple_sentences.txt", 'r').readlines()
    for line in sdata:
        if len(line.split()) >=6:    
            non_simplified_sentences.append(line.strip())

    sentences = non_simplified_sentences + simplified_sentences
    return sentences   

# Load data
def load_data_manf(path):
    non_simplified_sentences = []
    simplified_sentences = []

    data = open(path+"manifesto_simplified.txt", 'r').readlines()
    for line in data:
        line_val = json.loads(line)
        for item in line_val['simplified']:
            if len(item["text"].split()) >=6:   
                simplified_sentences.append(item['text'])

    sentences = simplified_sentences
    return sentences   


# Run embeddings and clustering
def run(args):
    path = PATH + args.path + "/"
    sentences = load_data_manf(path)
    embeddings = embedd_sentences(sentences)

    labels = run_kmeans(embeddings, args.n_clusters)
    
    assert(len(labels) == len(sentences) == len(embeddings))

    data_frame = pd.DataFrame({"sentence": sentences, "label": labels})
    data_frame.to_csv(path+"manifesto_clustered_numclusters_{}.csv".format(args.n_clusters), index=False)

# Main
def main():
    parser = argparse.ArgumentParser(description='Run K-Means clustering on the dataset')
    parser.add_argument('--path', type=str, default=PATH, help='Path to the dataset')
    parser.add_argument('--n_clusters', type=int, default=44, help='Number of clusters')
    args = parser.parse_args()

    run(args)

if __name__ == '__main__':
    main()
    