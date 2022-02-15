import json, pickle
from operator import le
import os, sys, time, math, random
import argparse
from typing_extensions import final

import numpy as np
import pandas as pd

from utils import DatasetManager
from utils import train, evaluate
from utils import asMinutes, timeSince

np.random.seed(0)
random.seed(0)


from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

import seaborn as sns
import matplotlib.pyplot as plt

import spacy   
from spacy.matcher import Matcher
from spacy.util import filter_spans
nlp = spacy.load('en_core_web_sm')

from itertools import combinations

# Load the embeddings of the specified dataset
def load_embeddings(fname1, fname2, flabel):

    PATH = 'paraphrase/data/'
    
    full_file_path1 = PATH + fname1 + ".pkl"
    full_file_path2 = PATH + fname2 + ".pkl"
    full_path_label = PATH + flabel + ".pkl"

    # Load full dataset with combined NLI pairs
    with open(full_file_path1, "rb") as em1:
        stored_data_1 = pickle.load(em1)
    with open(full_file_path2, "rb") as em2:
        stored_data_2 = pickle.load(em2)
    with open(full_path_label, "rb") as lbl:
        stored_labels = pickle.load(lbl)

    input_vectors1 = stored_data_1['embeddings']
    print(type(input_vectors1), input_vectors1.shape) # Shape (9076, 384)
    input_vectors2 = stored_data_2['embeddings']
    print(type(input_vectors2), input_vectors2.shape) # Shape (9076, 384)

    # product_of_vectors = np.einsum('ij,ij->i', input_vectors1, input_vectors2)[..., None]
    product_of_vectors = input_vectors1 * input_vectors2
    print(type(product_of_vectors), product_of_vectors.shape) # Shape (9076, 384)

    labels = np.array(stored_labels['labels'])
    print(type(labels), labels.shape)

    X_train, X_test, y_train, y_test = train_test_split(product_of_vectors, labels, 
                                        test_size=0.2, shuffle = True, random_state = 0)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    return X_train, X_test, y_train, y_test

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

# Train the model and take 20% of the training set as the dev/val set 
def run_model(X_train, X_test, y_train, y_test, out_file_train):

    clf = Ridge(alpha=1.0).fit(X_train, y_train)
    weights = clf.coef_

    preds = clf.predict(X_test)
    
    #print("Predictions for 100 are", preds[0:100])
    #print("Prediction probs for 100 are", pred_probs[0:100])

    print("Accuracy is:", clf.score(X_test, y_test))
    
    '''SAVE_PATH =  "paraphrase/figs/ridge_cm_train_" + out_file_train +".png"
    figure.savefig(SAVE_PATH)
    plt.close(figure)'''

    # SAVE MODEL FOR FUTURE USE (with training dataset name)
    MODEL_PATH = "paraphrase/saved_models/"

    filename = MODEL_PATH + "_ridge_" + out_file_train + ".sav"
    pickle.dump(clf, open(filename, 'wb'))

    return clf, weights

# Run the trained model on the test dataset, which the model has not seen
def evaluate_model(clf, weights, fname1, fname2, flabel, out_file_train, out_file_test):

    print("Testing on {} dataset, trained on {} dataset".format(out_file_train, out_file_test))

    if out_file_train == "full":
        print("Full means MPRC train set + NLI contradiction pairs.........")

    PATH = 'paraphrase/data/'
    
    full_file_path1 = PATH + fname1 + ".pkl"
    full_file_path2 = PATH + fname2 + ".pkl"
    full_path_label = PATH + flabel + ".pkl"

    # Load test dataset 
    with open(full_file_path1, "rb") as _em1:
        test_data_1 = pickle.load(_em1)
    with open(full_file_path2, "rb") as _em2:
        test_data_2 = pickle.load(_em2)
    with open(full_path_label, "rb") as _lbl:
        test_labels = pickle.load(_lbl)

    test_vectors1, test_vectors2 = test_data_1['embeddings'], test_data_2['embeddings']
    
    elem_prod = test_vectors1 * test_vectors2
    combined_test = elem_prod

    print(combined_test.shape)  
    t_labels = np.array(test_labels['labels'])
    print(t_labels.shape)    

    print("Metrics for test dataset......")       

    t_preds = clf.predict(combined_test) 
    print("Predictions for 10 are", t_preds[0:10])
    
    print("Accuracy for test set is:", clf.score(combined_test, t_labels)) 
    
    '''SAVE_PATH =  "paraphrase/figs/ridge_cm_train_" + out_file_train + "_test_" + out_file_test + ".png"   
    figure1.savefig(SAVE_PATH)
    plt.close(figure1)'''

    true_labels = t_labels
    print("Weights shape")

    print(weights.shape)

    y_hat = np.einsum("ij,ij,j->i", test_vectors1, test_vectors2, weights)

    return clf, combined_test, test_data_1, test_data_2, t_labels, y_hat


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train", help="train dataset specifier")
    parser.add_argument("-ev", "--eval", help="eval dataset specifier")
    parser.add_argument("-ts", "--test", help="test corpus dataset specifier")
   
    args = parser.parse_args()

    if args.train == "full":
        fname1 = "embeddings_1"
        fname2 = "embeddings_2"
        flabel = "labels"

    elif args.train == "paws":
        fname1 = "train_embeddings_paws1"
        fname2 = "train_embeddings_paws2"
        flabel = "train_labels_paws"
    
    elif args.train == "mprc":
        fname1 = "mprc_embeddings_1"
        fname2 = "mprc_embeddings_2"
        flabel = "mprc_labels"

    else:
        print("Invalid train dataset")
        exit()

    # Get the embeddings in the train and eval subsets
    X_train, X_test, y_train, y_test = load_embeddings(fname1, fname2, flabel)

    # Run the classifier model (logistic regression for now)
    classifier, weights = run_model(X_train, X_test, y_train, y_test, args.train)

    if args.eval == "mprc":
        eval_fname1 = "test_embeddings_1"
        eval_fname2 = "test_embeddings_2"
        eval_flabel = "test_labels"

    elif args.eval == "paws":
        eval_fname1 = "test_embeddings_paws1"
        eval_fname2 = "test_embeddings_paws2"
        eval_flabel = "test_labels_paws"
    else:
        print("Invalid test dataset")
        exit()

    if args.test == "corp1":
        test_fname = "test_corpus1"

    # Evaluate the model on the test dataset
    test_classifier, combined_vec, s_vec1, s_vec2, s_labels, s_preds = evaluate_model(classifier, weights, eval_fname1, 
                                                                eval_fname2, eval_flabel, args.train, args.eval)
    print(s_labels.shape)
    print(s_preds.shape)

    print(mean_squared_error(s_preds, s_labels))

    indices_0 = np.where(s_labels==0)
    indices_1 = np.where(s_labels==1)

    y_hat_true_label_0 = s_preds[indices_0]
    y_hat_true_label_1 = s_preds[indices_1]
    # print(y_hat_true_label_0.shape)
    # print(y_hat_true_label_1.shape)


    plt.figure(1)
    plt.hist(s_labels, bins='auto')  
    plt.title("Histogram of true_labels")
    plt.savefig("paraphrase/figs/hist_true_labels_mrpc.png",format="png")

    plt.figure(2)
    plt.hist(s_preds, bins='auto')  
    plt.title("Histogram of y_hat")
    plt.savefig("paraphrase/figs/hist_y_hat_mrpc.png",format="png")

    plt.figure(3)
    plt.hist(y_hat_true_label_1, bins='auto')  
    plt.title("Histogram of y_hat giveen true label is 1")
    plt.savefig("paraphrase/figs/hist_y_hat_tl_1mrpc.png",format="png")

    plt.figure(4)
    plt.hist(y_hat_true_label_0, bins='auto')  
    plt.title("Histogram of y_hat giveen true label is 0")
    plt.savefig("paraphrase/figs/hist_y_hat_tl_0mrpc.png",format="png")




if __name__ == '__main__':
    main()
