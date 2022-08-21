from __future__ import annotations
from scipy import stats
from tqdm import tqdm

import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from operator import le

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import spacy
import torch.nn
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from spacy.matcher import Matcher
from spacy.util import filter_spans

nlp = spacy.load("en_core_web_sm")


STS_SAVED_FILE_NAME = "sts_embeddings.pkl"
DATA_PATH = "paraphrase/test_corpora/"
EMBEDDING_PATH = "paraphrase/data/"
PATH = "paraphrase/figs/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
print(device, model)


def load_STS_English_data():

    text_a, text_b = [], []

    label_file = "STS2017.gs/STS.gs.track5.en-en.txt"
    with open(DATA_PATH + label_file) as f:
        labels = [float(line.strip()) for line in f]

    sentence_file = "STS2017.eval.v1.1/STS.input.track5.en-en.txt"
    with open(DATA_PATH + sentence_file) as f:
        for line in f:
            line = line.strip().split("\t")
            text_a.append(line[0])
            text_b.append(line[1])
    return text_a, text_b, labels


# Generate SBERT embeddings
def generate_embeddings(model, sent1, sent2, labels):
    sent1_embs = model.encode(sent1)
    sent2_embs = model.encode(sent2)
    labels = labels

    with open(DATA_PATH + STS_SAVED_FILE_NAME, "wb") as fOut_sts:

        pickle.dump(
            {
                "sentences1": sent1,
                "embeddings1": sent1_embs,
                "sentences2": sent2,
                "embeddings2": sent2_embs,
                "labels": labels,
            },
            fOut_sts,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    return None


# LOAD the weights of the trained logistic model
def load_saved_model(model_path):
    loaded_model = pickle.load(open(model_path, "rb"))
    return loaded_model


# LOAD embeddings from stored state
def load_embeddings(fname):
    with open(fname, "rb") as em:
        stored_data = pickle.load(em)
    return stored_data


# EVALUATE corpus sentence pairs
def evaluate_model(clf, vectors1, vectors2):

    print("Testing pairs on corpus above threshold")

    test_vectors1, test_vectors2 = vectors1, vectors2
    abs_diff = np.abs(test_vectors1 - test_vectors2)
    elem_prod = test_vectors1 * test_vectors2

    combined_test = np.concatenate(
        (test_vectors1, test_vectors2, abs_diff, elem_prod), axis=1
    )
    print(combined_test.shape)

    print("Metrics for test dataset......")

    t_preds = clf.predict(combined_test)
    t_pred_probs = clf.predict_proba(combined_test)

    print("Predictions for 10 are", t_preds[0:10])
    print("Prediction probs for 10 are", t_pred_probs[0:10])

    return clf, t_preds, t_pred_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-file", "--file", help="choose csv file for loading", default="sts"
    )
    parser.add_argument(
        "-sv", "--save", required=False, help="choose to generate sts embeddings"
    )

    args = parser.parse_args()

    if args.file == "sts":
        text_a, text_b, lbls = load_STS_English_data()
        print(len(text_a), len(text_b), len(lbls))
        print(text_a[0], text_b[0], lbls[0], sep="\n")

    if args.save:
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        generate_embeddings(model, text_a, text_b, lbls)
    else:
        loaded_data = load_embeddings(DATA_PATH + STS_SAVED_FILE_NAME)
        sent1, sent2 = loaded_data["sentences1"], loaded_data["sentences2"]
        embs1, embs2 = loaded_data["embeddings1"], loaded_data["embeddings2"]
        labels = loaded_data["labels"]

        print(len(sent1), len(sent2))
        print(embs1.shape, embs2.shape)
        print(len(labels))

    model_path = "paraphrase/saved_models/full.sav"

    saved_model = load_saved_model(model_path)
    print("loaded saved model....")

    model, pred, pred_probs = evaluate_model(saved_model, embs1, embs2)
    print(pred_probs.shape)

    count_0 = 0
    count_1 = 0

    for i in pred:
        if i == 0:
            count_0 += 1
        else:
            count_1 += 1

    print(count_0, count_1)

    rho_yes = stats.spearmanr(pred_probs[:, 1].tolist(), labels)
    rho_no = stats.spearmanr(pred_probs[:, 0].tolist(), labels)

    print("correlation with yes prob: ", rho_yes)
    print("correlation with no prob: ", rho_no)


if __name__ == "__main__":
    main()
