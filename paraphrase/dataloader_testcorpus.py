import json, pickle, argparse
import random
import math
from re import L

import torch 

from sentence_transformers import SentenceTransformer

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import numpy as np
from numpy import linalg as LA

from itertools import combinations

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

file = "paraphrase/test_corpora/source_corpus2.csv"
stored_file = "paraphrase/data/test_corpus1.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('all-MiniLM-L6-v2', device = device)
print(device, model)

# GET corpus as list of sentences
def get_corpus(full_file_path):

    data_file = pd.read_csv(full_file_path)
    list_of_sentences = data_file['text'].to_list()

    return list_of_sentences

#FILTER corpus based on indices
def filter_corpus_as_dataframe(full_file_path, list_of_indices):
    data_file = pd.read_csv(full_file_path)['text']
    df_new = data_file.iloc[list_of_indices]
    return df_new

# SLOW pair calculation from list of sentences
def get_pairs_of_sentences(sentences):
    
    sentence_pairs = list(combinations(sentences,2))

    sentence1 = sentence_pairs[:, 0]
    sentence2 = sentence_pairs[:, 1]

    return sentence1, sentence2

# FAST cosine pairwise function
def pairwise_cosine_sim_matrix(input_matrix):
    m = input_matrix
    # norm = (m * m).sum(0, keepdims=True) ** .5
    norm = LA.norm(m, axis = 1, keepdims = True)
    m_norm = m/norm; 
    similarity_matrix = m_norm @ m_norm.T 

    return similarity_matrix

# SAVE embeddings for next time
def generate_and_save_embeddings(sentences):
    
    list_of_sentences = sentences
    list_of_embeddings = model.encode(sentences)
    
    with open('paraphrase/data/test_corpus1.pkl', "wb") as fOut1:
        pickle.dump({'sentences': list_of_sentences, 'embeddings': list_of_embeddings}, fOut1, protocol=pickle.HIGHEST_PROTOCOL)

    return list_of_embeddings, list_of_sentences

# LOAD embeddings from stored state
def load_embeddings(fname):

    with open(fname, "rb") as em:
        stored_data = pickle.load(em)
    
    return stored_data

# FILTER for different thresholds, find mean, save mean cosine values
def filter_matrixes_by_threshold_get_mean(cos_matrix, threshold):

    thr= float(threshold)

    thresholds = []
    mean_values = []
    median_values = []
    num_elem = []

    for value in np.arange(thr, 0.15, -0.025):

        thresholds.append(value)
        # SET threshold for pairwise similarity
        masked_matrix = np.where(cos_matrix > value , 1, 0)
        # print(masked_matrix[0][0:15])
        indices_for_similar = np.where(masked_matrix==1)

        # Print first indices for entries above threshold
        # print(indices_for_similar[0][0:10])
        # print(indices_for_similar[1][0:10])

        values_above_threshold = cos_matrix[indices_for_similar[0], indices_for_similar[1]]
        print(values_above_threshold[0:10])
        mean_values.append(np.mean(values_above_threshold))
        median_values.append(np.median(values_above_threshold))
        num_elem.append(values_above_threshold.size / 2)

    return thresholds, mean_values, median_values, num_elem

# For one threshold, get all incides for satisfying pairs
def filter_for_single_threshold(cos_matrix, threshold):
    thr= float(threshold)

    masked_matrix = np.where(cos_matrix > thr , 1, 0)
    indices_for_similar = np.where(masked_matrix==1)
    print(indices_for_similar[0].shape)
    print(indices_for_similar[1].shape)

    return indices_for_similar[0], indices_for_similar[1]

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-dev", "--device", help="device specifier")
    parser.add_argument("-sv", "--save", help = "choose saved numpy cosine matrix")
    parser.add_argument("-thr", "--threshold", help = "theshold for filtering cosine sim")
    parser.add_argument("-plt", "--plot", help = "if set, plot for decreasing threshold values")

    args = parser.parse_args()

    print(args)

    if args.save:
        
        if args.device == "gpu":
            print("generating new embeddings........")
            sentences = get_corpus(file)
            print(sentences[0:5])
            list_of_embeddings, list_of_sentences = generate_and_save_embeddings(sentences)
            print(list_of_embeddings.shape)

        elif args.device == "cpu":
            print("loading stored embeddings........")
            stored_data = load_embeddings(stored_file)
            list_of_embeddings = stored_data['embeddings']
            # print(list_of_embeddings.shape)

            pair_cosine_matrix = pairwise_cosine_sim_matrix(list_of_embeddings)
            # print(pair_cosine_matrix.shape)
            # print(pair_cosine_matrix[0][0:15])

            # np.savetxt("paraphrase/figs/cosine_sim.csv", pair_cosine_matrix, delimiter=",")
            np.save("paraphrase/data/cosine_sim.npy", pair_cosine_matrix)
            np.save("paraphrase/data/cosine_sim_16.npy", pair_cosine_matrix.astype(np.float16))
   
    else:
        print("Loading from saved.....")
        loaded_pair_cosine_matrix = np.load("paraphrase/data/cosine_sim_16.npy")
        print(loaded_pair_cosine_matrix.shape)
        print(loaded_pair_cosine_matrix[0][0:15])

        # For varying thresholds, get the mean, median and number of sentence pairs
        if args.plot:

            thresholds, mean_values, median_values, num_elem = filter_matrixes_by_threshold_get_mean(loaded_pair_cosine_matrix, 
                                                                                                    args.threshold)
            plt.figure(1)
            plt.plot(thresholds, mean_values)
            plt.title("Threshold vs mean cosine similarity on satisfying indices")
            plt.xlabel("Threshold")
            plt.ylabel("Mean cosine similarity")
            plt.savefig("paraphrase/figs/threshold_cosine_mean.png",format="png")

            plt.figure(2)
            plt.plot(thresholds, median_values)
            plt.title("Threshold vs median cosine similarity on satisfying indices")
            plt.xlabel("Threshold")
            plt.ylabel("Median cosine similarity")
            plt.savefig("paraphrase/figs/threshold_cosine_median.png",format="png")


            plt.figure(3)
            plt.plot(thresholds, num_elem)
            plt.title("Threshold vs number of sentence pairs above threshold")
            plt.xlabel("Threshold")
            plt.ylabel("Number of sentence pairs above threshold")
            plt.savefig("paraphrase/figs/threshold_num.png",format="png")
        
        else:
            first_sentence_indices, second_sentence_indices = filter_for_single_threshold(loaded_pair_cosine_matrix,
                                                                                        args.threshold)
            print(first_sentence_indices[0:10])
            print(second_sentence_indices[0:10])

            '''df_new1 = filter_corpus_as_dataframe(file, first_sentence_indices.tolist())
            df_new2 = filter_corpus_as_dataframe(file, second_sentence_indices.tolist())
            df_new1.columns = ["sent1"]
            df_new1.columns = ["sent2"]
            print(df_new1.shape)
            print(df_new2.shape)'''

            #new_df = pd.concat([df_new1, df_new2], axis=1)
            #print(new_df.head())

            print("This is from numpy")
            stored_embeddings = load_embeddings(stored_file)

            sent_vectors1 = stored_embeddings['embeddings'][first_sentence_indices.tolist()]
            sent_vectors2 = stored_embeddings['embeddings'][second_sentence_indices.tolist()]

            print(sent_vectors1.shape)
            print(sent_vectors2.shape)

            '''sentences1 = np.asarray(stored_embeddings['sentences'])[first_sentence_indices.tolist()]
            sentences2 = np.asarray(stored_embeddings['sentences'])[second_sentence_indices.tolist()]

            print(sentences1.shape)
            print(sentences2.shape)'''

            ## TODO : Dump these vectors as a pickle file, they have O(n^2) pairs now.
            ## Not enough space to save these, save the list of indices instead

            '''with open('paraphrase/data/testcorpus_embeddings_sent1.pkl', "wb") as fOut1:
                pickle.dump({'embeddings': sent_vectors1}, fOut1, protocol=pickle.HIGHEST_PROTOCOL)

            with open('paraphrase/data/testcorpus_embeddings_sent2.pkl', "wb") as fOut2:
                pickle.dump({'embeddings': sent_vectors2}, fOut2, protocol=pickle.HIGHEST_PROTOCOL)'''

            np.save("paraphrase/data/sent1_indices.npy", first_sentence_indices)
            np.save("paraphrase/data/sent2_indices.npy", second_sentence_indices)


            #SAVE_PATH = "paraphrase/data/pairwise_corpus_on_thr_above" + args.threshold + ".csv" 
            #new_df.to_csv(SAVE_PATH)



if __name__ == '__main__':
    main()
        





