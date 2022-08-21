from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import hdbscan
import numpy as np
import pandas as pd
import umap
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV

PATH = "simplification_clustering/datasets/"

# Get embeddings


def embedd_sentences(sentences):

    sbert_model = "all-MiniLM-L6-v2"
    max_seq_length = 128
    embedder = SentenceTransformer(sbert_model)
    embedder.max_seq_length = max_seq_length
    corpus_embeddings = embedder.encode(
        sentences, batch_size=32, device="cuda", show_progress_bar=True
    )
    return corpus_embeddings


# Get clusters
def run_kmeans(X, n_clusters=44):

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=512).fit(X)
    labels = kmeans.labels_
    return labels


# Get clusters from HDBScan
def run_hdbscan(X, cluster_size, samples, reduced=False):

    if reduced:
        # Apply PCA to reduce to 64, then umap to reduce to 8 (should be faster)
        pca = PCA(n_components=64)
        X_pca = pca.fit_transform(X)

        umap_ = umap.UMAP(n_neighbors=10, min_dist=0.5, n_components=8)
        X_umap = umap_.fit_transform(X_pca)
        X = X_umap

    hdb = hdbscan.HDBSCAN(
        gen_min_span_tree=True, min_cluster_size=cluster_size, min_samples=samples
    ).fit(X)
    labels = hdb.labels_

    # Score over search space
    """hdb = hdbscan.HDBSCAN(gen_min_span_tree=True).fit(X)
    param_dist = {'min_samples': [15,30,50,100],
                'min_cluster_size':[10,50,100,250,500],
                'cluster_selection_method' : ['eom','leaf'],
                'metric' : ['euclidean']
                }

    validity_scorer = make_scorer(hdbscan.validity.validity_index,greater_is_better=True)

    n_iter_search = 20
    random_search = RandomizedSearchCV(hdb,
                                    param_distributions=param_dist,
                                    n_iter=n_iter_search,
                                    scoring=validity_scorer,
                                    random_state=0)
    random_search.fit(X)

    print(f"Best Parameters {random_search.best_params_}")
    print(f"DBCV score :{random_search.best_estimator_.relative_validity_}")"""

    return labels


# Load data
def load_data(path):
    non_simplified_sentences = []
    simplified_sentences = []

    data = open(path + "bbc_data_complex_splitted.txt", "r").readlines()
    for line in data:
        line_val = json.loads(line)
        for item in line_val["simplified"]:
            if len(item["text"].split()) >= 6:
                simplified_sentences.append(item["text"])

    sdata = open(path + "bbc_data_simple_sentences.txt", "r").readlines()
    for line in sdata:
        if len(line.split()) >= 6:
            non_simplified_sentences.append(line.strip())

    sentences = non_simplified_sentences + simplified_sentences
    return sentences


# Load data
def load_data_manf(path):
    non_simplified_sentences = []
    simplified_sentences = []

    data = open(path + "manifesto_simplified.txt", "r").readlines()
    for line in data:
        line_val = json.loads(line)
        for item in line_val["simplified"]:
            if len(item["text"].split()) >= 6:
                simplified_sentences.append(item["text"])

    sentences = simplified_sentences
    return sentences


# Load data
def load_data_manf_full_sent(path):
    non_simplified_sentences = []

    data = open(path + "manifesto_simplified.txt", "r").readlines()
    for line in data:
        line_val = json.loads(line)
        for item in line_val["simplified"]:
            if len(item["text"].split()) >= 6:
                non_simplified_sentences.append(line_val["original"])
                continue

    sentences = non_simplified_sentences
    return sentences


# Run embeddings and clustering
def run(args):
    path = PATH + args.path + "/"
    sentences = load_data_manf_full_sent(path)
    embeddings = embedd_sentences(sentences)

    if args.model == "kmeans":
        labels = run_kmeans(embeddings, args.n_clusters)
    elif args.model == "hdbscan":
        labels = run_hdbscan(
            embeddings, args.cluster_size, args.samples, reduced=args.reduction
        )

    assert len(labels) == len(sentences) == len(embeddings)

    data_frame = pd.DataFrame({"sentence": sentences, "label": labels})
    if args.model == "kmeans":
        data_frame.to_csv(
            path +
            "manifesto_clustered_numclusters_{}.csv".format(args.n_clusters),
            index=False,
        )
    elif args.model == "hdbscan":
        data_frame.to_csv(
            path
            + "manifesto_clustered_hdbscan_min_cluster_{}.csv".format(
                args.cluster_size
            ),
            index=False,
        )
        print(data_frame["label"].value_counts())


# Main
def main():
    parser = argparse.ArgumentParser(
        description="Run K-Means clustering on the dataset"
    )
    parser.add_argument("--path", type=str, default=PATH,
                        help="Path to the dataset")
    parser.add_argument(
        "--model", type=str, help="Choose clustering model", required=True
    )
    parser.add_argument(
        "--reduction", type=bool, default=False, help="Apply PCA and UMAP"
    )
    parser.add_argument("--cluster_size", type=int, help="Cluster size")
    parser.add_argument("--samples", type=int, help="Samples")
    args = parser.parse_args()

    print(args)

    run(args)


if __name__ == "__main__":
    main()
