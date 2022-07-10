from operator import ne
import os, json, pickle, sys, time
import argparse
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score
import hdbscan
from sklearn.decomposition import PCA
import umap

PATH = "simplification_clustering/datasets/"

# Get embeddings
def embedd_sentences(sentences):
    
    sbert_model = 'all-MiniLM-L6-v2'
    max_seq_length = 256
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
def load_data_manf(path):
    non_simplified_sentences = []
    
    data = open(path+"manifesto_simplified.txt", 'r').readlines()
    for line in data:
        line_val = json.loads(line)
        for item in line_val['simplified']:
            if len(item["text"].split()) >=6:   
                non_simplified_sentences.append(line_val['original'])
                continue

    sentences = non_simplified_sentences
    return sentences   

# Evaluate baseline
def evaluate(path,clustered_frame,nclusters):
    # Load map
    mapping_df_with_labels= pd.read_csv(path+"mapping_data.csv", )
    mapping_df_with_labels["original"].drop_duplicates(inplace=True)
   
    # Keep rows with unique original and sentences 
    clustered_frame = clustered_frame.drop_duplicates(subset=["sentence"])
    mapping_df_with_labels = mapping_df_with_labels.drop_duplicates(subset=["original"])

    print(mapping_df_with_labels.head(), mapping_df_with_labels.shape)
    print(clustered_frame.head(), clustered_frame.shape)

    # Get labels from mapping data
    mapping = {i:j for i,j in zip(mapping_df_with_labels["original"], mapping_df_with_labels["label1"])}

    max_cluster_label_dict = {}

    for i in range(nclusters):
        df_small = clustered_frame[clustered_frame.label == i]
        labels = [mapping[i] for i in df_small["sentence"]]
        counts = np.unique(labels, return_counts=True)

        # Check for empty cluster id
        if counts[1].size == 0:
            continue

        argmax = counts[1].argmax()
        print ("argmax", argmax, "num occurrences", counts[1][argmax])
        print ("label", counts[0][argmax])

        max_cluster_label_dict[i] = counts[0][argmax]

    print(max_cluster_label_dict)

    # Create a new column which is a copy of the label column 
    clustered_frame["true max label"] = clustered_frame["label"]
    # Replace every value in this column by its dictionary value using apply
    clustered_frame["true max label"] = clustered_frame["true max label"].apply(lambda x: max_cluster_label_dict[x])
    clustered_frame.to_csv(path+"manifesto_nonsimplfy_clustered_numclusters_{}.csv".format(nclusters), index=False)

    ## Compute cluster accuracy 
    mapping_df_with_labels.sort_values(by=["original"], inplace=True)
    mapping_df_with_labels['label1'] = mapping_df_with_labels['label1'].fillna("No label")

    clustered_frame.sort_values(by=["sentence"], inplace=True)

    targets = list(mapping_df_with_labels["label1"])    
    predictions = list(clustered_frame["true max label"])

    print ("Accuracy of numclusters = {}:".format(nclusters), accuracy_score(targets, predictions))



# Run embeddings and clustering
def run(args):
    path = PATH + args.path + "/"
    sentences = load_data_manf(path)
    embeddings = embedd_sentences(sentences)

    if args.model == "kmeans":
        labels = run_kmeans(embeddings, args.n_clusters)
    
    assert(len(labels) == len(sentences) == len(embeddings))

    data_frame = pd.DataFrame({"sentence": sentences, "label": labels})
    if args.model == "kmeans": 
        data_frame.to_csv(path+"manifesto_nonsimplfy_clustered_numclusters_{}.csv".format(args.n_clusters), index=False)

    ## Evaluate the clustering
    evaluate(path, data_frame, args.n_clusters)

# Main
def main():
    parser = argparse.ArgumentParser(description='Run K-Means clustering on the dataset')
    parser.add_argument('--path', type=str, default=PATH, help='Path to the dataset')
    parser.add_argument('--model', type=str, help='Choose clustering model', required=True)
    parser.add_argument('--n_clusters', type=int, help='Number of clusters', required=True)
    args = parser.parse_args()
    run(args)

    args = parser.parse_args()

    run(args)

if __name__ == '__main__':
    main()
    
