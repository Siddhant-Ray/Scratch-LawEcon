from __future__ import annotations

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
from nltk.tokenize import sent_tokenize
from spacy.matcher import Matcher
from spacy.util import filter_spans

nlp = spacy.load("en_core_web_sm")

from tqdm import tqdm

PATH = "paraphrase/figs/"

# Test corpus
stored_file = "paraphrase/data/test_corpus1.pkl"
stored_file_bbc = "paraphrase/data/test_corpus_bbc.pkl"
stored_file_trump = "paraphrase/data/test_corpus_trump.pkl"
stored_file_custom = "paraphrase/data/test_corpus_custom.pkl"
stored_file_memsum = "paraphrase/data/test_corpus_memsum.pkl"
stored_indices_path = "paraphrase/data/"
data_file = "paraphrase/test_corpora/source_corpus2.csv"
data_file_bbc = "paraphrase/test_corpora/bbc_data.csv"
data_file_trump = "paraphrase/test_corpora/trump_archive.csv"
data_file_custom = "paraphrase/test_corpora/custom_train_fromjson.csv"


def read_csv(path):
    df = pd.read_csv(path)
    return df


# Filter dataframe by a threshold paraphrase probability
def filter_dataframe(df, threshold):
    threshold = float(threshold)
    df = df.loc[df["prob_score"] >= threshold]
    df = df.drop(
        columns=["indirect words sent1", "count of verbs sent1", "verbs in sent1"]
    )
    df = df.drop(
        columns=["indirect words sent2", "count of verbs sent2", "verbs in sent2"]
    )
    df = df.rename(columns={"prob_score": "paraphrase_probability"})
    return df


# Save the filtered cv file
def save_filtered_csv(df, save_path, train_set, threshold):
    save_path = (
        PATH
        + "trained_on_"
        + train_set
        + "_trainset_"
        + "mprc_fulltestset"
        + "_filtered_paraprob_greater_than"
        + str(threshold)
        + ".csv"
    )
    df.to_csv(save_path, index=False)


# Compute pairwise cosine similarities on the two sentence sets
def cosine_similarities_on_train_set(data_path, save_path, trainset):

    PATH = data_path

    full_file_path1 = PATH + "embeddings_1" + ".pkl"
    full_file_path2 = PATH + "embeddings_2" + ".pkl"
    full_path_label = PATH + "labels" + ".pkl"

    with open(full_file_path1, "rb") as em1:
        stored_data_1 = pickle.load(em1)
    with open(full_file_path2, "rb") as em2:
        stored_data_2 = pickle.load(em2)
    with open(full_path_label, "rb") as lbl:
        stored_labels = pickle.load(lbl)

    v1 = stored_data_1["embeddings"]
    v2 = stored_data_2["embeddings"]
    labels = np.array(stored_labels["labels"])

    product_of_vectors = np.einsum("ij,ij->i", v1, v2)[..., None]
    normedv1 = (v1 * v1).sum(axis=1) ** 0.5
    normedv2 = (v2 * v2).sum(axis=1) ** 0.5

    inverse_prod_norms = np.reciprocal(normedv1 * normedv2).reshape(-1, 1)
    cosine_similarites = product_of_vectors * inverse_prod_norms

    df1 = pd.DataFrame(stored_data_1["sentences"], columns=["sent1"])
    df1.index = np.arange(1, len(df1) + 1)
    # print(df1.head())
    df2 = pd.DataFrame(stored_data_2["sentences"], columns=["sent2"])
    df2.index = np.arange(1, len(df2) + 1)
    # print(df2.head())
    df3 = pd.DataFrame(cosine_similarites, columns=["cosine_sim"])
    df3.index = np.arange(1, len(df3) + 1)
    # print(df3.head())
    df4 = pd.DataFrame(labels, columns=["true_labels"])
    df4.index = np.arange(1, len(df4) + 1)
    # print(df4.head())

    final_df = pd.concat([df1, df2, df3, df4], axis=1)
    print(final_df.head())

    SAVE_PATH = save_path + "cosine_similarities_on" + trainset + "_trainset" + ".csv"
    final_df.to_csv(SAVE_PATH)

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


# LOAD numpy indices for pairs above a threshold
def load_indices(path):
    sent1_path = path + "sent1_indices.npy"
    sent2_path = path + "sent2_indices.npy"

    sent1_indices = np.load(sent1_path)
    sent2_indices = np.load(sent2_path)

    return sent1_indices, sent2_indices


# LOAD numpy indices for pairs above a threshold
def load_unique_indices(path):
    sent1_path = path + "sent1_indices_noequal.npy"
    sent2_path = path + "sent2_indices_noequal.npy"

    sent1_indices = np.load(sent1_path)
    sent2_indices = np.load(sent2_path)

    return sent1_indices, sent2_indices


# LOAD numpy indices for pairs above a threshold for bbc
def load_indices_bbc(path):
    sent1_path = path + "sent1_indices_bbc.npy"
    sent2_path = path + "sent2_indices_bbc.npy"

    sent1_indices = np.load(sent1_path)
    sent2_indices = np.load(sent2_path)

    return sent1_indices, sent2_indices


# LOAD numpy indices for pairs above a threshold for bbc
def load_unique_indices_bbc(path):
    sent1_path = path + "sent1_indices_noequal_bbc.npy"
    sent2_path = path + "sent2_indices_noequal_bbc.npy"

    sent1_indices = np.load(sent1_path)
    sent2_indices = np.load(sent2_path)

    return sent1_indices, sent2_indices


# EVALUATE corpus sentence pairs (subset above threshold)
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


# EVALUATE corpus sentence pairs, less memory usage but SLOW
def evaluate_model_slow(clf, vectors):

    para_probs = []
    print("Evaluating para probs on pair wise sentences")

    for row in tqdm(vectors):
        vector1 = np.array(len(vectors) * [row])
        vector2 = vectors
        assert vector1.shape[0] == vector2.shape[0]

        abs_diff = np.abs(vector1 - vector2)
        elem_prod = vector1 * vector2
        combined_test = np.concatenate((vector1, vector2, abs_diff, elem_prod), axis=1)
        # print(combined_test.shape)
        # print("Metrics for test dataset......")
        t_preds = clf.predict(combined_test)
        t_pred_probs = clf.predict_proba(combined_test)
        # print(t_pred_probs.shape)
        para_probs = para_probs + t_pred_probs[:, 1].tolist()

    t_pred_probs = np.array(para_probs)
    print("Prediction probs for 10 are", t_pred_probs[0:10])
    print("Shape of paraprobs is", t_pred_probs.shape)

    return clf, t_pred_probs


# FILTER corpus based on indices
def filter_corpus_as_dataframe(full_file_path, list_of_indices):
    data_file = pd.read_csv(full_file_path)["text"]
    df_new = data_file.iloc[list_of_indices]
    return df_new


# GET BBC corpus
def get_bbc_corpus(full_file_path):
    data_file = pd.read_csv(full_file_path)
    list_of_paras = data_file["transcript"]
    list_of_sentences = list_of_paras.str.split(".")
    df_sentences = pd.DataFrame({"sents": list_of_sentences})
    new_df = df_sentences.explode("sents", ignore_index=True)
    return list_of_paras, list_of_sentences, new_df["sents"]


# GET BBC corpus sentence wise via Spacy
def get_bbc_corpus_nltk(full_file_path):
    data_file = pd.read_csv(full_file_path)
    new_df = pd.DataFrame({"transcript": data_file.transcript})
    new_df["tokenized_sents"] = new_df.apply(
        lambda row: sent_tokenize(row["transcript"]), axis=1
    )
    new_df = new_df.drop(columns=["transcript"])
    new_df = new_df.explode("tokenized_sents", ignore_index=True)
    return new_df


# FILTER sentences on list of indices with cosine sim threshold
def filter_bbc_corpus(dataframe, list_of_indices):
    new_df = dataframe.iloc[list_of_indices]
    return new_df


# Get all verbs using a Spacy based function
def get_verbs(input_sentence):

    sentence = input_sentence
    pattern = [
        {"POS": "VERB", "OP": "?"},
        {"POS": "ADV", "OP": "*"},
        {"POS": "AUX", "OP": "*"},
        {"POS": "VERB", "OP": "+"},
    ]

    # instantiate a Matcher instance
    matcher = Matcher(nlp.vocab)
    matcher.add("Verb phrase", [pattern])

    doc = nlp(sentence)
    # call the matcher to find matches
    matches = matcher(doc)
    spans = [doc[start:end] for _, start, end in matches]

    return filter_spans(spans)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-file", "--file", help="choose csv file for loading")
    parser.add_argument(
        "-th", "--threshold", help="threshold to filter cosine similarities"
    )
    parser.add_argument("-sv", "--save", help="used saved indices or not")
    parser.add_argument("-dt", "--data", help="choose to take the bbc corpus")
    parser.add_argument(
        "-noeq", "--noequal", help="choose whether to include same sentences as pairs"
    )
    parser.add_argument("-k", "--knumelem", help="how many top/ bottom k to select")

    args = parser.parse_args()

    if args.file == "full":
        name1 = "full"
        name2 = "mprc"
    elif args.file == "mprc":
        name1 = "mprc"
        name2 = "mprc"
    elif args.file == "paws":
        name1 = "paws"
        name2 = "paws"
    else:
        print("Invalid CSV file, exiting.....\n")
        exit()

    full_path = PATH + "paraphr_trainset_" + name1 + "_testset_" + name2 + ".csv"

    data_frame = read_csv(full_path)
    print(data_frame.head())

    filtered_df = filter_dataframe(data_frame, args.threshold)
    print(filtered_df.head())

    # Save the file
    save_filtered_csv(filtered_df, full_path, args.file, args.threshold)

    DATA_PATH = "paraphrase/data/"
    SAVE_PATH = "paraphrase/figs/"

    cosine_similarities_on_train_set(DATA_PATH, SAVE_PATH, args.file)

    model_path = "paraphrase/saved_models/full.sav"

    saved_model = load_saved_model(model_path)
    model_coeffs = saved_model.coef_
    model_biases = saved_model.intercept_
    # print(model_coeffs.shape)
    # print(model_biases.shape)

    if args.data == "bbc":
        save_name = args.data
        print("From {}".format(save_name))
        stored_data = load_embeddings(stored_file_bbc)

    elif args.data == "trump":
        save_name = args.data
        print("From {}".format(save_name))
        stored_data = load_embeddings(stored_file_trump)

    elif args.data == "custom":
        save_name = args.data
        print("From {}".format(save_name))
        stored_data = load_embeddings(stored_file_custom)

    elif args.data == "memsum":
        save_name = args.data
        print("From {}".format(save_name))
        stored_data = load_embeddings(stored_file_memsum)

    else:
        save_name = "bigcorpus"
        print("From {}".format(save_name))
        stored_data = load_embeddings(stored_file)

    list_of_embeddings = stored_data["embeddings"]
    print(list_of_embeddings.shape)

    if args.noequal and args.data == "bbc":
        sent1_indices, sent2_indices = load_unique_indices_bbc(stored_indices_path)
        print("Loading pairs {} without equality........".format(save_name))
        print(sent1_indices.shape, sent2_indices.shape)

    elif args.data == "bbc":
        sent1_indices, sent2_indices = load_indices_bbc(stored_indices_path)
        print("Loading pairs {} with equality........".format(save_name))
        print(sent1_indices.shape, sent2_indices.shape)

    elif args.noequal and args.data != "bbc":
        sent1_indices, sent2_indices = load_unique_indices(stored_indices_path)
        print("Loading pairs {} without equality........".format(save_name))
        print(sent1_indices.shape, sent2_indices.shape)
    elif args.data == "trump":
        pass
    elif args.data == "custom":
        pass
    elif args.data == "memsum":
        pass
    else:
        sent1_indices, sent2_indices = load_indices(stored_indices_path)
        print("Loading pairs {} with equality........".format(save_name))
        print(sent1_indices.shape, sent2_indices.shape)

    if save_name == "trump" or save_name == "custom" or args.data == "memsum":
        sent_vectors = list_of_embeddings
    elif save_name == "bbc" or save_name == "bigcorpus":
        ### Run the model on the sentence pairs on the big corpus
        sent_vectors_1 = list_of_embeddings[sent1_indices]
        sent_vectors_2 = list_of_embeddings[sent2_indices]
    else:
        pass

    if args.save:
        print("Running model on {} dataset".format(save_name))
        if save_name == "trump" or save_name == "custom" or save_name == "memsum":
            model, probs = evaluate_model_slow(saved_model, sent_vectors)
            print(probs[0:10])
            print(probs.shape)
            para_probs = probs
        else:
            model, preds, probs = evaluate_model(
                saved_model, sent_vectors_1, sent_vectors_2
            )
            print(probs[0:10])
            print(probs[:, 1].shape)
            para_probs = probs[:, 1]

        # plt.figure(1)
        # plt.hist(para_probs, bins='auto')
        # plt.title("Histogram of para_probs for {}".format(save_name))

        if args.noequal:
            # plt.savefig("paraphrase/figs/hist_para_probs_0.5_noequal_thresh_{}.png".format(save_name),format="png")
            np.save(
                "paraphrase/data/para_probs_noequal_{}.npy".format(save_name),
                para_probs,
            )
        else:
            # plt.savefig("paraphrase/figs/hist_para_probs_0.5_thresh_{}.png".format(save_name),format="png")
            np.save("paraphrase/data/para_probs_{}.npy".format(save_name), para_probs)

    else:
        if args.noequal:
            print(
                "Loading para probs from {} without equal pairs.......".format(
                    save_name
                )
            )
            para_probs = np.load(
                "paraphrase/data/para_probs_noequal_{}.npy".format(save_name)
            )
        else:
            print(
                "Loading para probs from {} with equal pairs.......".format(save_name)
            )
            para_probs = np.load("paraphrase/data/para_probs_{}.npy".format(save_name))

        # Set the value of k for top k/bottom k pairs
        k_value = int(args.knumelem)
        bottom_k_indices = np.argsort(para_probs)[:k_value]
        top_k_indices = np.argsort(-para_probs)[:k_value]

        # print(bottom_k_indices)

        if args.data == "bbc":
            # list_of_paras, list_of_sentences, sent_dataframe = get_bbc_corpus(data_file_bbc)
            # USE SPACYYYY.....
            sent_dataframe = get_bbc_corpus_nltk(data_file_bbc)
            df_sent1 = filter_bbc_corpus(sent_dataframe, sent1_indices.tolist())
            df_sent2 = filter_bbc_corpus(sent_dataframe, sent2_indices.tolist())

        elif args.data != "bbc":
            df_sent1 = filter_corpus_as_dataframe(data_file, sent1_indices.tolist())
            df_sent2 = filter_corpus_as_dataframe(data_file, sent2_indices.tolist())

        print("Para probs size sanity check from {}".format(save_name))
        print(para_probs.shape)
        print(df_sent1.head())
        print(df_sent2.head())

        topk_sent1 = df_sent1.iloc[top_k_indices.tolist()]
        topk_sent2 = df_sent2.iloc[top_k_indices.tolist()]

        # Top k
        # print(topk_sent1.head(), topk_sent2.head())
        # print(topk_sent1.shape, topk_sent2.shape)

        bottomk_sent1 = df_sent1.iloc[bottom_k_indices.tolist()]
        bottomk_sent2 = df_sent2.iloc[bottom_k_indices.tolist()]

        # Bottom k
        # print(bottomk_sent1.head(), bottomk_sent2.head())
        # print(bottomk_sent1.shape, bottomk_sent2.shape)

        topk_sent1.reset_index(drop=True, inplace=True)
        topk_sent2.reset_index(drop=True, inplace=True)
        bottomk_sent1.reset_index(drop=True, inplace=True)
        bottomk_sent2.reset_index(drop=True, inplace=True)

        df_probs_top = pd.DataFrame(para_probs[top_k_indices])[0]
        # print(df_probs_top.head())
        # print(df_probs_top.shape)
        df_probs_top.columns = ["para_prob"]
        new_df = pd.concat([topk_sent1, topk_sent2, df_probs_top], axis=1)
        new_df.columns = ["sent1", "sent2", "para_probs"]
        if args.noequal:
            new_df.to_csv(
                SAVE_PATH + "top_{}_noequal_{}.csv".format(k_value, save_name)
            )
        else:
            new_df.to_csv(SAVE_PATH + "top_{}_{}.csv".format(k_value, save_name))

        df_probs_bottom = pd.DataFrame(para_probs[bottom_k_indices])[0]
        # print(df_probs_bottom.head())
        # print(df_probs_bottom.shape)
        df_probs_bottom.columns = ["para_prob"]
        new_df = pd.concat([bottomk_sent1, bottomk_sent2, df_probs_bottom], axis=1)
        new_df.columns = ["sent1", "sent2", "para_probs"]
        if args.noequal:
            new_df.to_csv(
                SAVE_PATH + "bottom_{}_noequal_{}.csv".format(k_value, save_name)
            )
        else:
            new_df.to_csv(SAVE_PATH + "bottom_{}_{}.csv".format(k_value, save_name))


if __name__ == "__main__":
    main()
