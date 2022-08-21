from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from re import M

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

PATH = "simplification_clustering/datasets/"

# Load mapping data
def load_data_manf_map(path):
    non_simplified_sentences = []
    simplified_sentences = []

    mapping_df = pd.DataFrame(columns=["original", "simplified"])

    data = open(path + "manifesto_simplified.txt", "r").readlines()
    for line in data:
        line_val = json.loads(line)
        for idx, item in enumerate(line_val["simplified"]):
            if len(item["text"].split()) >= 6:
                simplified_sentences.append(item["text"])
                non_simplified_sentences.append(line_val["original"])

    mapping_df["original"] = non_simplified_sentences
    mapping_df["simplified"] = simplified_sentences

    return mapping_df


# Load actual labels
def load_actual_labels(path):

    data = pd.read_csv(path + "manifesto_map.txt", sep="\t", on_bad_lines="skip")
    data.columns = ["original", "label1", "label2", "original2"]

    return data


# Map between data frames
def map_df(mapping_df, actual_labels):

    actual_labels.set_index("original", inplace=True)
    label1_dict = actual_labels.to_dict()["label1"]
    label2_dict = actual_labels.to_dict()["label2"]

    for key in label1_dict.keys():
        mapping_df.loc[mapping_df["original"] == key, "label1"] = label1_dict[key]
        mapping_df.loc[mapping_df["original"] == key, "label2"] = label2_dict[key]

    return mapping_df


def run(args):
    path = PATH + args.path + "/"

    if not args.load:
        mapping_df = load_data_manf_map(path)
        actual_labels = load_actual_labels(path)
        print(mapping_df.head())
        mapping_df_with_labels = map_df(mapping_df, actual_labels)
        mapping_df_with_labels.to_csv(path + "mapping_data.csv", index=False)

    else:
        # Load map
        mapping_df_with_labels = pd.read_csv(path + "mapping_data.csv")

        # Load clustered files
        clustered_frame = pd.read_csv(
            path + "manifesto_clustered_numclusters_{}.csv".format(args.n_clusters)
        )
        # Sort clustered frame by label ascending
        clustered_frame.sort_values(by=["label"], inplace=True)

        # Get labels from mapping data
        mapping = {
            i: j
            for i, j in zip(
                mapping_df_with_labels["simplified"], mapping_df_with_labels["label1"]
            )
        }

        max_cluster_label_dict = {}

        for i in range(args.n_clusters):
            df_small = clustered_frame[clustered_frame.label == i]
            labels = [mapping[i] for i in df_small["sentence"]]
            counts = np.unique(labels, return_counts=True)

            # Check for empty cluster id
            if counts[1].size == 0:
                continue

            argmax = counts[1].argmax()
            print("argmax", argmax, "num occurrences", counts[1][argmax])
            print("label", counts[0][argmax])

            max_cluster_label_dict[i] = counts[0][argmax]

        print(max_cluster_label_dict)

        # Create a new column which is a copy of the label column
        clustered_frame["true max label"] = clustered_frame["label"]
        # Replace every value in this column by its dictionary value using apply
        clustered_frame["true max label"] = clustered_frame["true max label"].apply(
            lambda x: max_cluster_label_dict[x]
        )
        clustered_frame.to_csv(
            path + "manifesto_clustered_numclusters_{}.csv".format(args.n_clusters),
            index=False,
        )

        ## Compute cluster accuracy
        mapping_df_with_labels.sort_values(by=["simplified"], inplace=True)
        mapping_df_with_labels["label1"] = mapping_df_with_labels["label1"].fillna(
            "No label"
        )

        clustered_frame.sort_values(by=["sentence"], inplace=True)

        targets = list(mapping_df_with_labels["label1"])
        predictions = list(clustered_frame["true max label"])

        assert list(clustered_frame["sentence"]) == list(
            mapping_df_with_labels["simplified"]
        )
        assert len(targets) == len(predictions)

        print(
            "Accuracy of numclusters = {}:".format(args.n_clusters),
            accuracy_score(targets, predictions),
        )


# Main
def main():
    parser = argparse.ArgumentParser(description="Evaluate dataset")
    parser.add_argument("--path", type=str, default=PATH, help="Path to the dataset")
    parser.add_argument("--load", type=bool, default=None, help="Path to the dataset")
    parser.add_argument(
        "--n_clusters", type=int, default=None, help="Number of clusters"
    )
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
