from __future__ import annotations

import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

# Load full dataset with combined NLI pairs
with open("paraphrase/data/embeddings_1.pkl", "rb") as em1:
    stored_data_1 = pickle.load(em1)
with open("paraphrase/data/embeddings_2.pkl", "rb") as em2:
    stored_data_2 = pickle.load(em2)
with open("paraphrase/data/labels.pkl", "rb") as lbl:
    stored_labels = pickle.load(lbl)

# Load only mprc dataset
with open("paraphrase/data/mprc_embeddings_1.pkl", "rb") as _em1:
    stored_data_mprc_1 = pickle.load(_em1)
with open("paraphrase/data/mprc_embeddings_2.pkl", "rb") as _em2:
    stored_data_mprc_2 = pickle.load(_em2)
with open("paraphrase/data/mprc_labels.pkl", "rb") as _lbl:
    stored_labels_mprc = pickle.load(_lbl)

print(len(stored_data_mprc_1["embeddings"]),
      type(stored_data_mprc_1["embeddings"]))
print(len(stored_data_mprc_2["embeddings"]),
      type(stored_data_mprc_2["embeddings"]))
print(len(stored_labels_mprc["labels"]), type(stored_labels_mprc["labels"]))

sent_vecs_mprc_1, sent_vecs1 = (
    stored_data_mprc_1["embeddings"],
    stored_data_1["embeddings"],
)
sent_vecs_mprc_2, sent_vecs2 = (
    stored_data_mprc_2["embeddings"],
    stored_data_2["embeddings"],
)
target_labels_mprc, target_labels = (
    stored_labels_mprc["labels"],
    stored_labels["labels"],
)

"""TSNE_embedded1 = TSNE(n_components = 2, perplexity = 40, n_jobs = -1, random_state = 0).fit_transform(sent_vecs1)
print(TSNE_embedded1.shape)

df = pd.DataFrame({'X': TSNE_embedded1[:, 0], 'Y': TSNE_embedded1[:, 1], 'Label': target_labels})
df.to_csv('paraphrase/plots/tSNE_values_vecs1.csv', index=False)

TSNE_embedded2 = TSNE(n_components = 2, perplexity = 40, n_jobs = -1, random_state = 0).fit_transform(sent_vecs2)
print(TSNE_embedded2.shape)

df = pd.DataFrame({'X': TSNE_embedded2[:, 0], 'Y': TSNE_embedded2[:, 1], 'Label': target_labels})
df.to_csv('paraphrase/plots/tSNE_values_vecs2.csv', index=False)"""

# Plot functions

files = [
    "paraphrase/plots/tSNE_values_mprc_vecs1.csv",
    "paraphrase/plots/tSNE_values_mprc_vecs2.csv",
    "paraphrase/plots/tSNE_values_vecs1.csv",
    "paraphrase/plots/tSNE_values_vecs2.csv",
]

map_classes = {"yes": 1, "no": 0}

for _file in files:
    file_name_to_save = _file.split("/")[-1].replace(".csv", "") + ".png"
    df = pd.read_csv(_file)
    fig = plt.figure(figsize=(15, 15))
    out = plt.scatter(df["X"], df["Y"], 10, c=df["Label"])
    # cbar = plt.colorbar(out, ticks = np.array([1,0]))
    plt.legend(*out.legend_elements())
    plt.title("tSNE on pre-classified output vs ground truth labels", size=30)
    plt.xticks(size=25)
    plt.yticks(size=25)
    plt.savefig("paraphrase/plots" + "/" + file_name_to_save)
