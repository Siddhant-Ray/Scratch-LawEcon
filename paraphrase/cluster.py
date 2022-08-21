from __future__ import annotations

import argparse
import email
import json
import math
import os
import pickle
import random
import sys
import time
from collections import Counter
from collections import defaultdict
from operator import le
from tkinter import N

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from yellowbrick.cluster import KElbowVisualizer

PATH = "paraphrase/data/"
embedding_file_bbc = "paraphrase/data/test_corpus_bbc.pkl"
embedding_file_trump = "paraphrase/data/test_corpus_trump.pkl"
embedding_file_custom = "paraphrase/data/test_corpus_custom.pkl"
embedding_file_memsum = "paraphrase/data/test_corpus_memsum.pkl"

# LOAD numpy indices for pairs above a threshold for bbc
# We want equality pairs for the distance matrix, else 0 for (i,i)
# may give us wrong results


def load_indices_bbc(path):
    sent1_path = path + "sent1_indices_bbc.npy"
    sent2_path = path + "sent2_indices_bbc.npy"

    sent1_indices = np.load(sent1_path)
    sent2_indices = np.load(sent2_path)

    return sent1_indices, sent2_indices


# LOAD unique numpy indices for pairs above a threshold for bbc
# No (i,i) case, defined just in case
def load_unique_indices_bbc(path):
    sent1_path = path + "sent1_indices_noequal_bbc.npy"
    sent2_path = path + "sent2_indices_noequal_bbc.npy"

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
    _, ax = plt.subplots()  # Create a new figure
    model = clf
    visualizer = KElbowVisualizer(model, k=(4, 12), title=title, ax=ax)
    visualizer.fit(input_data)
    visualizer.show(outpath=out_path)


# COMPUTE agglomerative clustering
def custom_agglomerative_clustering(input_data, n_clusters, linkage):
    model = AgglomerativeClustering(
        n_clusters, affinity="precomputed", linkage=linkage)
    clusters = model.fit(input_data)
    labels = clusters.labels_
    return clusters, labels


# COMPUTE spectral clustering
def custom_spectral_clustering(input_data, n_clusters, affinity):
    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity=affinity,
        assign_labels="discretize",
        random_state=42,
        n_jobs=-1,
    )
    clusters = model.fit(input_data)
    labels = clusters.labels_
    return clusters, labels


#  COMPUTE DBSCAN clustering
def custom_dbscan_clustering(input_data, metric):
    model = DBSCAN(metric=metric, n_jobs=-1)
    clusters = model.fit(input_data)
    labels = clusters.labels_
    return clusters, labels


# LOAD embeddings and sentences from stored state
def load_embeddings(fname):
    with open(fname, "rb") as em:
        stored_data = pickle.load(em)
    return stored_data


# slightly adapted from https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    return linkage_matrix


def get_clusters_from_linkage_matrix(linkage_matrix, depth_factor):
    # depth_of_tree basically determines how many merging steps we allow,
    # len(linkage_matrix // 4 ) seems to yield decent results,
    # i.e., ca. 500 clusters)
    # depth_of_tree=len(linkage_matrix) // 2
    # depth_of_tree=len(linkage_matrix)

    depth_of_tree = int(len(linkage_matrix) - 10)
    print("depth of tree")
    print(depth_of_tree)

    clusters = defaultdict(set)
    c = len(linkage_matrix) + 1
    n = len(linkage_matrix)
    for i in linkage_matrix[:depth_of_tree]:
        a, b = int(i[0]), int(i[1])
        if a > n:
            clusters[c] = clusters[a]
            del clusters[a]
        else:
            clusters[c].add(a)
        if b > n:
            clusters[c].update(list(clusters[b]))
            del clusters[b]
        else:
            clusters[c].add(b)
        if len(clusters[c]) == 1:
            pass
        c += 1
    return clusters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", "--data", help="choose to take the bbc corpus")
    parser.add_argument(
        "-noeq", "--noequal", help="choose whether to include same sentences as pairs"
    )
    parser.add_argument(
        "-clf", "--classifier", help="choose classifier for kelbow", default=KMeans()
    )
    parser.add_argument("-mtx", "--matrix", help="specify the matrix of input")
    parser.add_argument(
        "-vis", "--visualize", help="decide if kelbow should be plotted"
    )
    parser.add_argument(
        "-mod", "--model", help="clustering model selection", required=True
    )
    parser.add_argument(
        "-link", "--linkage", help="decide linkage for agglomerative clustering"
    )
    parser.add_argument(
        "-aff", "--affinity", help="decide affinity for spectral clustering"
    )
    parser.add_argument("-met", "--metric",
                        help="decide metric for DBSCAN clustering")
    parser.add_argument(
        "-mtype", "--matrix_type", help="specify distance or similarity matrix"
    )
    parser.add_argument(
        "-nclus", "--nclusters", help="specify the number of clusters for agglomerative"
    )
    parser.add_argument("-load", "--load",
                        help="create csv from saved cluster labels")
    parser.add_argument("-sent", "--sentences",
                        help="cluster raw sentence vector")
    parser.add_argument(
        "-cust", "--custom", help="use custom clustering based on agglomerative"
    )
    parser.add_argument("-dpt", "--depth",
                        help="specifiy depth of merge factor")

    args = parser.parse_args()

    if args.data == "bbc":
        if args.noequal:
            sent1_indices, sent2_indices = load_unique_indices_bbc(PATH)
            file = PATH + "para_probs_noequal_{}.npy".format(args.data)
            para_probs = load_paraphrase_probs(file)
            print(
                "Loaded BBC pairs indices from saved, {} ones .....".format(
                    args.noequal
                )
            )
        else:
            sent1_indices, sent2_indices = load_indices_bbc(PATH)
            file = PATH + "para_probs_{}.npy".format(args.data)
            para_probs = load_paraphrase_probs(file)
            print("Loaded BBC pairs indices from saved with (i,i) pairs .....")

        print("indices shape", sent1_indices.shape, sent2_indices.shape)
        print("para_probs shape", para_probs.shape)

        sentence_embeddings = load_embeddings(embedding_file_bbc)["embeddings"]
        sentences = load_embeddings(embedding_file_bbc)["sentences"]
        print("Sentence vectors loaded from {} .....".format(args.data))
        print("Shape of sentence vectors", sentence_embeddings.shape)

        tuples_of_indices = zip(sent1_indices.tolist(), sent2_indices.tolist())
        # Shape of matrix should be sentence X sentences
        # This matrix has the paraphrase probabilities for certain indices
        # which indices have a paraphrase probability , with cosine sim
        # > 0.5 between the pair

        matrix_init = np.zeros(
            (sentence_embeddings.shape[0], sentence_embeddings.shape[0])
        )
        print("Empty matrix.......")
        print(matrix_init[0], matrix_init.shape)

        for ix, index in enumerate(tuples_of_indices):
            matrix_init[index] = para_probs[ix]

        print("Similarity matrix filled with paraprobs.......")
        print(matrix_init[0], matrix_init.shape)

        ones_matrix = np.ones(
            (sentence_embeddings.shape[0], sentence_embeddings.shape[0])
        )
        print("Empty ones matrix.......")
        print(ones_matrix[0], ones_matrix.shape)

        dist_matrix = ones_matrix - matrix_init
        print("Distance matrix.......")
        print(dist_matrix[0], dist_matrix.shape)

    elif args.data == "trump":

        file = PATH + "para_probs_{}.npy".format(args.data)
        para_probs = load_paraphrase_probs(file)
        print(
            "Loaded {} pairs indices from saved with (i,i) pairs .....".format(
                args.data
            )
        )
        print("para_probs shape", para_probs.shape)

        sentence_embeddings = load_embeddings(
            embedding_file_trump)["embeddings"]
        sentences = load_embeddings(embedding_file_trump)["sentences"]
        print("Sentence vectors loaded from {} .....".format(args.data))
        print("Shape of sentence vectors", sentence_embeddings.shape)

        ones_matrix = np.ones(
            (sentence_embeddings.shape[0], sentence_embeddings.shape[0])
        )
        print("All ones matrix.......")
        print(ones_matrix[0], ones_matrix.shape)

        print("Similarity matrix filled with paraprobs.......")
        sim_matrix = para_probs.reshape(
            (sentence_embeddings.shape[0], sentence_embeddings.shape[0])
        )
        print(sim_matrix[0], sim_matrix.shape)

        dist_matrix = ones_matrix - sim_matrix
        matrix_init = sim_matrix
        print("Distance matrix.......")
        print(dist_matrix[0], dist_matrix.shape)

    elif args.data == "custom":
        file = PATH + "para_probs_{}.npy".format(args.data)
        para_probs = load_paraphrase_probs(file)
        print(
            "Loaded {} pairs indices from saved with (i,i) pairs .....".format(
                args.data
            )
        )
        print("para_probs shape", para_probs.shape)

        sentence_embeddings = load_embeddings(
            embedding_file_custom)["embeddings"]
        sentences = load_embeddings(embedding_file_custom)[
            "sentences"].tolist()
        labels = load_embeddings(embedding_file_custom)["labels"].tolist()
        print("Sentence vectors loaded from {} .....".format(args.data))
        print("Shape of sentence vectors", sentence_embeddings.shape)
        print("No of labels", len(labels))
        print("No of sentences", len(sentences))

        assert len(sentences) == len(labels)

        ones_matrix = np.ones(
            (sentence_embeddings.shape[0], sentence_embeddings.shape[0])
        )
        print("All ones matrix.......")
        print(ones_matrix[0], ones_matrix.shape)

        print("Similarity matrix filled with paraprobs.......")
        sim_matrix = para_probs.reshape(
            (sentence_embeddings.shape[0], sentence_embeddings.shape[0])
        )
        print(sim_matrix[0], sim_matrix.shape)

        dist_matrix = ones_matrix - sim_matrix
        matrix_init = sim_matrix
        print("Distance matrix.......")
        print(dist_matrix[0], dist_matrix.shape)

    elif args.data == "memsum":
        file = PATH + "para_probs_{}.npy".format(args.data)
        para_probs = load_paraphrase_probs(file)
        print(
            "Loaded {} pairs indices from saved with (i,i) pairs .....".format(
                args.data
            )
        )
        print("para_probs shape", para_probs.shape)

        sentence_embeddings = load_embeddings(
            embedding_file_memsum)["embeddings"]
        sentences = load_embeddings(embedding_file_memsum)["sentences"]
        print("Sentence vectors loaded from {} .....".format(args.data))
        print("Shape of sentence vectors", sentence_embeddings.shape)

        ones_matrix = np.ones(
            (sentence_embeddings.shape[0], sentence_embeddings.shape[0])
        )
        print("All ones matrix.......")
        print(ones_matrix[0], ones_matrix.shape)

        print("Similarity matrix filled with paraprobs.......")
        sim_matrix = para_probs.reshape(
            (sentence_embeddings.shape[0], sentence_embeddings.shape[0])
        )
        print(sim_matrix[0], sim_matrix.shape)

        dist_matrix = ones_matrix - sim_matrix
        matrix_init = sim_matrix
        print("Distance matrix.......")
        print(dist_matrix[0], dist_matrix.shape)

    if args.classifier == "kmeans":
        model = KMeans(random_state=42)

    if args.matrix == "sentences":
        input_data = sentence_embeddings
        title = "Run on sentence embeddings"
        out_path = "paraphrase/figs/kelbow_kmeans_allsentences.png"
    elif args.matrix == "paraprobs":
        input_data = para_probs.reshape((-1, 1))
        title = "Run on para probs"
        out_path = "paraphrase/figs/kelbow_kmeans_paraprobs.png"

    if args.visualize:
        kelbow_visualize(input_data, model, title, out_path)

    if args.matrix_type == "sim":
        input_distance_matrix = matrix_init
    elif args.matrix_type == "dist":
        input_distance_matrix = dist_matrix
    else:
        print("input matrix type not selected")
        exit()

    if args.custom:
        model = AgglomerativeClustering(
            distance_threshold=0,
            affinity="precomputed",
            n_clusters=None,
            linkage=args.linkage,
        )
        model.fit(input_distance_matrix)
        np.save(
            "paraphrase/data/{}_matrix_{}.npy".format(
                args.matrix_type, args.data),
            input_distance_matrix,
        )
        linkage_matrix = plot_dendrogram(model, truncate_mode="level", p=3)
        clusters = get_clusters_from_linkage_matrix(linkage_matrix, args.depth)

        max_len = max(len(i) for i in clusters.values())
        index = [i for i, j in clusters.items() if len(j) == max_len][0]

        print("depth factor is ", args.depth)
        print(
            "num sentences appearing in clusters",
            sum(len(i) for i in clusters.values()),
        )
        print("n clusters", len(clusters), "max length", max_len)

        tokenized_sents = pd.DataFrame(sentences, columns=["tokenized_sents"])
        print(tokenized_sents.head())

        out = []
        for i, j in clusters.items():
            if len(j) > 5:
                for index in j:
                    out.append(
                        (tokenized_sents.iloc[index].tokenized_sents, i))
        df = pd.DataFrame(out, columns=["tokenized_sents", "cluster"])

        if args.data == "custom":
            true_labels = [labels[sentences.index(value[0])] for value in out]
            df["true_label"] = true_labels
            # print(true_labels[0:10])
            df["max_cluster_label"] = ""

            cluster_ids = set(df["cluster"])
            # print(cluster_ids)
            for value in cluster_ids:
                labels_for_cluster = df.loc[df["cluster"] == value]
                list_of_labels = labels_for_cluster["true_label"].tolist()
                max_label = max(list_of_labels, key=list_of_labels.count)
                # print(max_label)
                # print(labels_for_cluster.index)
                df["max_cluster_label"].iloc[labels_for_cluster.index] = max_label
                # print(df.iloc[labels_for_cluster.index])

        df = df.sort_values(by=["cluster"], ascending=False)
        reversed_cols = df.columns.tolist()[::-1]
        df = df[reversed_cols]
        print(df.head())
        df.to_csv(
            "paraphrase/figs/agglo_{}_custom_{}_sorted_dfactor_{}.csv".format(
                args.linkage, args.data, args.depth
            ),
            index=False,
        )

    else:
        # From the kelblow plots, we have k = 7
        n_clusters = int(args.nclusters)

        if not args.load:
            print("generating clusters....")

            if args.model == "agglo":

                print("Linkage method used is {}".format(args.linkage))
                clustered_model, labels = custom_agglomerative_clustering(
                    input_distance_matrix, n_clusters, args.linkage
                )
                print("Labels generated for agglomerative ......")

            elif args.model == "spectral":

                print("Affinity method used is {}".format(args.affinity))
                clustered_model, labels = custom_spectral_clustering(
                    input_distance_matrix, n_clusters, args.affinity
                )
                print("Labels generated for spectral ......")

            elif args.model == "dbscan":

                print("Metric used is {}".format(args.affinity))
                clustered_model, labels = custom_dbscan_clustering(
                    input_distance_matrix, args.metric
                )
                print("Labels generated for dbscan ......")

            else:
                print("specify model....")
                exit()

            print(labels.shape)
            print(labels[0:10])
            print("set of labels.....")
            print(set(labels.tolist()))

            np.save(
                "paraphrase/data/{}_labels_{}_{}_mtype_{}_nclusters_{}.npy".format(
                    args.model,
                    args.data,
                    args.linkage,
                    args.matrix_type,
                    str(n_clusters),
                ),
                labels,
            )

        elif args.load:
            print("loading clusters....")
            sentences = load_embeddings(embedding_file_bbc)["sentences"]
            print(sentences.shape)
            print(sentences[0:5])

            df = pd.DataFrame(sentences)
            df_2 = df
            numbers = [16, 32, 64, 128, 256, 512, 1024]

            for num in numbers:
                # df["{} clusters".format(num)] = np.load("paraphrase/data/agglo_labels_{}_{}_mtype_{}_nclusters_{}.npy".format(args.data,
                # args.linkage, args.matrix_type, str(num)))

                df_2["{} clusters".format(num)] = np.load(
                    "paraphrase/data/sent_vecs_agglo_labels_{}_{}_nclusters_{}.npy".format(
                        args.data, args.linkage, str(num)
                    )
                )
            # print(df.head())
            # df.to_csv("paraphrase/figs/agglo_{}_linkage_clustered.csv".format(args.linkage), index=False)
            df_2.to_csv(
                "paraphrase/figs/sent_vecs_agglo_{}_linkage_clustered.csv".format(
                    args.linkage
                ),
                index=False,
            )


if __name__ == "__main__":
    main()
