import json, pickle, argparse
import random, os
import math, string
from re import L

import torch 

from sentence_transformers import SentenceTransformer

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import numpy as np
from numpy import linalg as LA

from itertools import combinations
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

import spacy   
from spacy.matcher import Matcher
from spacy.util import filter_spans
from spacy.lang.en import English
nlp = spacy.load('en_core_web_sm')

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

file = "paraphrase/test_corpora/source_corpus2.csv"
file_bbc = "paraphrase/test_corpora/bbc_data.csv"
file_trump = "paraphrase/test_corpora/trump_archive.csv"
file_custom = "paraphrase/test_corpora/custom_train_fromjson.csv"
stored_file = "paraphrase/data/test_corpus1.pkl"
stored_file_bbc = "paraphrase/data/test_corpus_bbc.pkl"
stored_file_trump = "paraphrase/data/test_corpus_trump.pkl"
stored_file_custom = "paraphrase/data/test_corpus_custom.pkl"

folder_memsum = "paraphrase/test_corpora/extracted_archive/"
stored_file_memsum = "paraphrase/data/test_corpus_memsum.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('all-MiniLM-L6-v2', device = device)
print(device, model)

# GET corpus as list of sentences
def get_corpus(full_file_path):
    data_file = pd.read_csv(full_file_path)
    list_of_sentences = data_file['text'].to_list()
    return list_of_sentences

# GET the BBC corpus and also split into sentences, return 
# the list of split sentences

def get_bbc_corpus(full_file_path):
    data_file = pd.read_csv(full_file_path)
    list_of_paras = data_file['transcript']
    list_of_sentences = list_of_paras.str.split(".")
    df_sentences = pd.DataFrame({'sents':list_of_sentences})
    new_df = df_sentences.explode('sents', ignore_index=True)
    return list_of_paras, list_of_sentences, new_df['sents']

# NLTK based tokenizer, not the best (try Spacy)
def get_bbc_corpus_nltk(full_file_path):
    data_file = pd.read_csv(full_file_path)
    new_df = pd.DataFrame({"transcript":data_file.transcript})
    new_df['tokenized_sents'] = new_df.apply(lambda row: sent_tokenize(row['transcript']), axis=1)
    new_df = new_df.drop(columns=['transcript'])
    new_df = new_df.explode('tokenized_sents', ignore_index=True)
    return new_df

# Spacy based tokenizer
def custom_spacy_tokenizer(paragraph):
    _nlp = English()
    _nlp.add_pipe("sentencizer")
    doc = _nlp(paragraph)
    list_of_sents = [sent.text for sent in doc.sents]
    return list_of_sents

# NLTK based tokenizer, not the best (try Spacy)
def get_bbc_corpus_spacy(full_file_path):
    data_file = pd.read_csv(full_file_path)
    new_df = pd.DataFrame({"transcript":data_file.transcript})
    new_df['tokenized_sents'] = new_df.apply(lambda row: custom_spacy_tokenizer(row['transcript']), axis=1)
    new_df = new_df.drop(columns=['transcript'])
    new_df = new_df.explode('tokenized_sents', ignore_index=True)
    return new_df

# NLTK based trumnp corpus loader
def get_trump_corpus_nltk(full_file_path):
    data_file = pd.read_csv(full_file_path)
    new_df = pd.DataFrame({"doc":data_file.doc})
    new_df = new_df
    return new_df

# GET corpus for custom json derived csv
def get_custom_corpus(full_file_path):
    data_file = pd.read_csv(full_file_path)
    return data_file

# Get all verbs using a Spacy based function
def get_verbs(input_sentence):
    
    sentence = input_sentence
    pattern = [{'POS': 'VERB', 'OP': '?'},
            {'POS': 'ADV', 'OP': '*'},
            {'POS': 'AUX', 'OP': '*'},
            {'POS': 'VERB', 'OP': '+'}]

    # instantiate a Matcher instance
    matcher = Matcher(nlp.vocab)
    matcher.add("Verb phrase", [pattern])

    doc = nlp(sentence) 
    # call the matcher to find matches 
    matches = matcher(doc)
    spans = [doc[start:end] for _, start, end in matches]

    return (filter_spans(spans))

# Get sentence length
def get_sentence_length(input_sentence):
    _list = word_tokenize(input_sentence)
    return len(_list)

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

# SAVE embeddings for the BBC dataset
def generate_and_save_embeddings_bbc(sentences):

    list_of_sentences = sentences
    list_of_embeddings = model.encode(sentences)
    with open('paraphrase/data/test_corpus_bbc.pkl', "wb") as fOut1:
        pickle.dump({'sentences': list_of_sentences, 'embeddings': list_of_embeddings}, fOut1, protocol=pickle.HIGHEST_PROTOCOL)
    return list_of_embeddings, list_of_sentences

# SAVE embeddings for the trump dataset
def generate_and_save_embeddings_trump(sentences):

    list_of_sentences = sentences
    list_of_embeddings = model.encode(sentences)
    with open('paraphrase/data/test_corpus_trump.pkl', "wb") as fOut1:
        pickle.dump({'sentences': list_of_sentences, 'embeddings': list_of_embeddings}, fOut1, protocol=pickle.HIGHEST_PROTOCOL)
    return list_of_embeddings, list_of_sentences

 # SAVE embeddings for the custom dataset
def generate_and_save_embeddings_custom(data_frame):

    list_of_sentences = data_frame["texts"]
    list_of_labels = data_frame["labels"]
    list_of_embeddings = model.encode(list_of_sentences)
    with open('paraphrase/data/test_corpus_custom.pkl', "wb") as fOut1:
        pickle.dump({'sentences': list_of_sentences, 'embeddings': list_of_embeddings, 'labels': list_of_labels},
                                                 fOut1, protocol=pickle.HIGHEST_PROTOCOL)
    return list_of_embeddings, list_of_sentences, list_of_labels   


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

    masked_matrix = np.where(cos_matrix >= thr , 1, 0)
    indices_for_similar = np.where(masked_matrix==1)
    print(indices_for_similar[0].shape)
    print(indices_for_similar[1].shape)

    return indices_for_similar[0], indices_for_similar[1]

# Get combination of memsum extracted summaries
def get_memsum_corpus(folder):
    list_of_files = sorted(os.listdir(folder))
    list_of_all_sentences = []
    for file in list_of_files:
        if file.endswith(".txt"):
            print(file)
            text = open(folder + file)
            lines = text.readlines()
            for line in lines:
                list_of_all_sentences.append(line.rstrip("\n"))
            text.close()
            
    # print(list_of_all_sentences)
    return list_of_all_sentences

# SAVE embeddings for the trump dataset
def generate_and_save_embeddings_memsum(sentences):

    list_of_sentences = sentences
    list_of_embeddings = model.encode(sentences)
    with open('paraphrase/data/test_corpus_memsum.pkl', "wb") as fOut1:
        pickle.dump({'sentences': list_of_sentences, 'embeddings': list_of_embeddings}, fOut1, protocol=pickle.HIGHEST_PROTOCOL)
    return list_of_embeddings, list_of_sentences

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-dev", "--device",
                        help="device specifier")
    parser.add_argument("-sv", "--save",
                        help = "choose saved numpy cosine matrix")
    parser.add_argument("-thr", "--threshold",
                        help = "theshold for filtering cosine sim")
    parser.add_argument("-plt", "--plot",
                        help = "if set, plot for decreasing threshold values")
    parser.add_argument("-dt", "--data",
                        help = "choose to take the bbc corpus")

    args = parser.parse_args()

    print(args)

    if args.save:
        
        if args.device == "gpu":
            if args.data == "bbc":
                print("generating new embeddings from {} ........".format(args.data))
                sentences = get_bbc_corpus_nltk(file_bbc)
                print(sentences.head(), sentences.shape)
                sentences['sent_verbs'] = sentences['tokenized_sents'].apply(lambda row: get_verbs(row))
                sentences['sent_len'] = sentences['tokenized_sents'].apply(lambda row: get_sentence_length(row))
                sentences['verb_num']= sentences['sent_verbs'].str.len()
                sentences = sentences[sentences.sent_len > 5]
                sentences = sentences[sentences.sent_len <= 20]
                sentences = sentences[sentences.verb_num <= 2]
                sentences = sentences.reset_index(drop=True)

                print(sentences.head(), sentences.shape)
                list_of_embeddings, list_of_sentences = generate_and_save_embeddings_bbc(sentences["tokenized_sents"])
                print(list_of_embeddings.shape)

            elif args.data == "trump":
                print("generating new embeddings from {} ........".format(args.data))
                sentences = get_trump_corpus_nltk(file_trump)
                print(sentences.head(), sentences.shape)
                sentences['sent_verbs'] = sentences['doc'].apply(lambda row: get_verbs(row))
                sentences['sent_len'] = sentences['doc'].apply(lambda row: get_sentence_length(row))
                sentences['verb_num']= sentences['sent_verbs'].str.len()
                sentences = sentences[sentences.sent_len > 5]
                sentences = sentences[sentences.verb_num <= 2]
                sentences = sentences.reset_index(drop=True)

                print(sentences.head(), sentences.shape)
                list_of_embeddings, list_of_sentences = generate_and_save_embeddings_trump(sentences["doc"].to_list())
                print(list_of_embeddings.shape)

            elif args.data == "custom":
                print("generating new embeddings from {} ........".format(args.data))
                data_frame = get_custom_corpus(file_custom)
                print(data_frame.head(), data_frame.shape)
                list_of_embeddings, list_of_sentences, list_of_labels = generate_and_save_embeddings_custom(data_frame)
                print(list_of_embeddings.shape)

            elif args.data == "memsum":
                print("generating new embeddings from {} ........".format(args.data))
                input_sentences = get_memsum_corpus(folder_memsum)
                list_of_embeddings, list_of_sentences = generate_and_save_embeddings_memsum(input_sentences)
                print(list_of_embeddings.shape)

            else:
                print("generating new embeddings for big corpus........")
                sentences = get_corpus(file)
                print(sentences[0:5])
                list_of_embeddings, list_of_sentences = generate_and_save_embeddings(sentences)
                print(list_of_embeddings.shape)

        elif args.device == "cpu":
            if args.data == "bbc":
                print("loading stored embeddings from {} ........".format(args.data))
                stored_data = load_embeddings(stored_file_bbc)
            elif args.data == "trump":
                print("loading stored embeddings from {} ........".format(args.data))
                stored_data = load_embeddings(stored_file_trump)
            elif args.data == "custom":
                print("loading stored embeddings from {} ........".format(args.data))
                stored_data = load_embeddings(stored_file_custom) 
            elif args.data == "memsum":
                print("loading stored embeddings from {} ........".format(args.data))
                stored_data = load_embeddings(stored_file_memsum)     
            else:
                print("loading stored embeddings from big corpus ........")
                stored_data = load_embeddings(stored_file)
            
            list_of_embeddings = stored_data['embeddings']
            print(list_of_embeddings.shape)

            pair_cosine_matrix = pairwise_cosine_sim_matrix(list_of_embeddings)
            print(pair_cosine_matrix.shape)
            print(pair_cosine_matrix[0][0:15])

            # np.savetxt("paraphrase/figs/cosine_sim.csv", pair_cosine_matrix, delimiter=",")
            if args.data == "bbc":
                np.save("paraphrase/data/cosine_sim_bbc.npy", pair_cosine_matrix)
                np.save("paraphrase/data/cosine_sim_16_bbc.npy", pair_cosine_matrix.astype(np.float16))

            elif args.data == "trump":
                np.save("paraphrase/data/cosine_sim_trump.npy", pair_cosine_matrix)
                np.save("paraphrase/data/cosine_sim_16_trump.npy", pair_cosine_matrix.astype(np.float16))
            elif args.data == "custom":
                np.save("paraphrase/data/cosine_sim_custom.npy", pair_cosine_matrix)
                np.save("paraphrase/data/cosine_sim_16_custom.npy", pair_cosine_matrix.astype(np.float16))
            elif args.data == "memsum":
                np.save("paraphrase/data/cosine_sim_memsum.npy", pair_cosine_matrix)
                np.save("paraphrase/data/cosine_sim_16_memsum.npy", pair_cosine_matrix.astype(np.float16))
            else:
                np.save("paraphrase/data/cosine_sim_bigcorpus.npy", pair_cosine_matrix)
                np.save("paraphrase/data/cosine_sim_16_bigcorpus.npy", pair_cosine_matrix.astype(np.float16))
   
    else:
        if args.data == "bbc":
            print("Loading cosine matrix from saved from {} .....".format(args.data))
            loaded_pair_cosine_matrix = np.load("paraphrase/data/cosine_sim_16_bbc.npy")
        else:
            print("Loading cosine matrix from saved from big corpus .....")
            loaded_pair_cosine_matrix = np.load("paraphrase/data/cosine_sim_16_bigcorpus.npy")
        print(loaded_pair_cosine_matrix.shape)
        print(loaded_pair_cosine_matrix[0][0:15])

        # For varying thresholds, get the mean, median and number of sentence pairs
        if args.plot:

            if args.data == "bbc":
                save_name = args.data
            else:
                save_name = "bigcorpus"

            thresholds, mean_values, median_values, num_elem = filter_matrixes_by_threshold_get_mean(loaded_pair_cosine_matrix, 
                                                                                                    args.threshold)
            plt.figure(1)
            plt.plot(thresholds, mean_values)
            plt.title("Threshold vs mean cosine similarity on satisfying indices on {} data".format(save_name), fontsize = 10, y = 1.1)
            plt.xlabel("Threshold")
            plt.ylabel("Mean cosine similarity")
            plt.savefig("paraphrase/figs/threshold_cosine_mean_{}.png".format(save_name),format="png")

            plt.figure(2)
            plt.plot(thresholds, median_values)
            plt.title("Threshold vs median cosine similarity on satisfying indices on {} data".format(save_name), fontsize = 10, y = 1.1)
            plt.xlabel("Threshold")
            plt.ylabel("Median cosine similarity")
            plt.savefig("paraphrase/figs/threshold_cosine_median_{}.png".format(save_name),format="png")


            plt.figure(3)
            plt.plot(thresholds, num_elem)
            plt.title("Threshold vs number of sentence pairs above threshold on {} data".format(save_name), fontsize = 10, y = 1.1)
            plt.xlabel("Threshold")
            plt.ylabel("Number of sentence pairs above threshold")
            plt.savefig("paraphrase/figs/threshold_num_{}.png".format(save_name),format="png")
        
        else:

            if args.data == "bbc":
                save_name = args.data
            else:
                save_name = "bigcorpus"
            
            first_sentence_indices, second_sentence_indices = filter_for_single_threshold(loaded_pair_cosine_matrix,
                                                                                        args.threshold)
            print(first_sentence_indices.shape, second_sentence_indices.shape)                                                                            
            print(first_sentence_indices[0:10])
            print(second_sentence_indices[0:10])

            first_sentence_indices_no_equal = np.where(first_sentence_indices != second_sentence_indices)
            second_sentence_indices_no_equal = np.where(second_sentence_indices != first_sentence_indices)

            print(first_sentence_indices_no_equal[0].shape)
            print(first_sentence_indices[first_sentence_indices_no_equal[0]][0:10])
            print(second_sentence_indices_no_equal[0].shape)
            print(second_sentence_indices[second_sentence_indices_no_equal[0]][0:10])
            

            '''df_new1 = filter_corpus_as_dataframe(file, first_sentence_indices.tolist())
            df_new2 = filter_corpus_as_dataframe(file, second_sentence_indices.tolist())
            df_new1.columns = ["sent1"]
            df_new2.columns = ["sent2"]
            print(df_new1.shape)
            print(df_new2.shape)'''

            #new_df = pd.concat([df_new1, df_new2], axis=1)
            #print(new_df.head())

            print("This is from numpy")

            if args.data == "bbc":
                stored_embeddings = load_embeddings(stored_file_bbc)
            else:
                stored_embeddings = load_embeddings(stored_file)

            sent_vectors1 = stored_embeddings['embeddings'][first_sentence_indices.tolist()]
            sent_vectors2 = stored_embeddings['embeddings'][second_sentence_indices.tolist()]

            print(sent_vectors1.shape)
            print(sent_vectors2.shape)
            
            ## TODO : Dump these vectors as a pickle file, they have O(n^2) pairs now.
            ## Not enough space to save these, save the list of indices instead

            np.save("paraphrase/data/sent1_indices_{}.npy".format(save_name), first_sentence_indices)
            np.save("paraphrase/data/sent2_indices_{}.npy".format(save_name), second_sentence_indices)
            np.save("paraphrase/data/sent1_indices_noequal_{}.npy".format(save_name), first_sentence_indices[first_sentence_indices_no_equal[0]])
            np.save("paraphrase/data/sent2_indices_noequal_{}.npy".format(save_name), second_sentence_indices[second_sentence_indices_no_equal[0]])

            #SAVE_PATH = "paraphrase/data/pairwise_corpus_on_thr_above" + args.threshold + ".csv" 
            #new_df.to_csv(SAVE_PATH)



if __name__ == '__main__':
    main()
        





