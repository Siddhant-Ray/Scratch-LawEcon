import json, pickle
from operator import le
import os, sys, time, math, random
import argparse

import numpy as np
import pandas as pd

PATH = "paraphrase/figs/"

# Test corpus
stored_file = "paraphrase/data/test_corpus1.pkl"

def read_csv(path):
    df = pd.read_csv(path)
    return df

# Filter dataframe by a threshold paraphrase probability
def filter_dataframe(df, threshold):
    threshold = float(threshold)
    df = df.loc[df['prob_score'] >= threshold]
    df = df.drop(columns=['indirect words sent1','count of verbs sent1','verbs in sent1'])
    df = df.drop(columns=['indirect words sent2','count of verbs sent2','verbs in sent2'])
    df = df.rename(columns={'prob_score': 'paraphrase_probability'})
    return df 

# Save the filtered cv file
def save_filtered_csv(df, save_path, train_set, threshold):
    save_path = PATH + "trained_on_" + train_set + "_trainset_" + "mprc_fulltestset" + "_filtered_paraprob_greater_than" + str(threshold) + ".csv"
    df.to_csv(save_path, index = False)

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

    v1 = stored_data_1['embeddings']
    v2 = stored_data_2['embeddings']
    labels = np.array(stored_labels['labels'])

    product_of_vectors = np.einsum('ij,ij->i', v1, v2)[..., None]
    normedv1 = (v1*v1).sum(axis=1)**0.5
    normedv2 = (v2*v2).sum(axis=1)**0.5

    inverse_prod_norms = np.reciprocal(normedv1 * normedv2).reshape(-1,1)
    cosine_similarites = product_of_vectors * inverse_prod_norms

    df1 = pd.DataFrame(stored_data_1['sentences'], columns=['sent1'])
    df1.index = np.arange(1, len(df1)+1)
    #print(df1.head())
    df2 = pd.DataFrame(stored_data_2['sentences'], columns=['sent2'])
    df2.index = np.arange(1, len(df2)+1)
    #print(df2.head())
    df3 = pd.DataFrame(cosine_similarites, columns=['cosine_sim'])
    df3.index = np.arange(1, len(df3)+1)
    #print(df3.head())
    df4= pd.DataFrame(labels, columns=['true_labels'])
    df4.index = np.arange(1, len(df4)+1)
    #print(df4.head())

    final_df = pd.concat([df1, df2, df3, df4], axis=1)
    print(final_df.head())

    SAVE_PATH = save_path + "cosine_similarities_on" + trainset + "_trainset" + ".csv" 
    final_df.to_csv(SAVE_PATH)

    return None 

# LOAD the weights of the trained logistic model
def load_saved_model(model_path):
    loaded_model = pickle.load(open(model_path, 'rb'))
    return loaded_model

# LOAD embeddings from stored state
def load_embeddings(fname):
    with open(fname, "rb") as em:
        stored_data = pickle.load(em)
    
    return stored_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-file", "--file", help="choose csv file for loading")
    parser.add_argument("-th", "--threshold", help="threshold to filter cosine similarities")
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

    full_path = PATH + "paraphr_trainset_" + name1 + "_testset_" + name2 +  ".csv"

    data_frame = read_csv(full_path)
    print(data_frame.head())

    filtered_df = filter_dataframe(data_frame, args.threshold)
    print(filtered_df.head())

    # Save the file 
    save_filtered_csv(filtered_df, full_path, args.file, args.threshold)

    DATA_PATH = 'paraphrase/data/'
    SAVE_PATH = 'paraphrase/figs/'

    cosine_similarities_on_train_set(DATA_PATH, SAVE_PATH, args.file)

    model_path = "paraphrase/saved_models/full.sav"

    saved_model = load_saved_model(model_path)
    model_coeffs = saved_model.coef_
    model_biases = saved_model.intercept_
    print(model_coeffs.shape)
    print(model_biases.shape)

    stored_data = load_embeddings(stored_file)
    list_of_embeddings = stored_data['embeddings']
    print(list_of_embeddings.shape)
    


if __name__== '__main__':
    main()

