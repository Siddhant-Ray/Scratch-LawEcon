import json, pickle
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

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

import spacy   
from spacy.matcher import Matcher
from spacy.util import filter_spans
nlp = spacy.load('en_core_web_sm')

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

    abs_diff_vectors = np.abs(input_vectors1 - input_vectors2)
    print(type(abs_diff_vectors), abs_diff_vectors.shape) # Shape (9076, 384)

    # product_of_vectors = np.einsum('ij,ij->i', input_vectors1, input_vectors2)[..., None]
    product_of_vectors = input_vectors1 * input_vectors2
    print(type(product_of_vectors), product_of_vectors.shape) # Shape (9076, 384)

    input_combined_vectors1 = np.concatenate((input_vectors1, 
                            input_vectors2, abs_diff_vectors,product_of_vectors), axis = 1) # Shape (9076, 1536)

    input_combined_vectors2 = np.concatenate((input_vectors2, 
                            input_vectors1, abs_diff_vectors,product_of_vectors), axis = 1) # Shape (9076, 1536)

    input_combined_vectors_all = np.concatenate((input_combined_vectors1, input_combined_vectors2), axis = 0) # Shape (18152, 1536)                       


    print(type(input_combined_vectors_all), input_combined_vectors_all.shape)
    labels = np.array(stored_labels['labels'])
    labels_all = np.concatenate([labels] * 2, axis=0)
    print(type(labels_all), labels_all.shape)


    X_train, X_test, y_train, y_test = train_test_split(input_combined_vectors_all, labels_all, 
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

    clf = LogisticRegression(penalty='l2',random_state=0, max_iter=1000,
                        n_jobs = -1).fit(X_train, y_train)

    preds = clf.predict(X_test)
    pred_probs = clf.predict_proba(X_test)

    #print("Predictions for 100 are", preds[0:100])
    #print("Prediction probs for 100 are", pred_probs[0:100])

    print("Accuracy is:", clf.score(X_test, y_test))
    print("F1score is: ", f1_score(y_test, preds,  average=None))

    # Form and print confusion matrix, plot heatmap
    c_matrix = confusion_matrix(y_test, preds, labels=[0, 1], normalize = "true")
    print(c_matrix)

    df_cm = pd.DataFrame(c_matrix, index = [0, 1] ,columns = [0, 1])
    matrix = sns.heatmap(df_cm, annot=True, cmap='Blues')
    #plt.figure()
    figure = matrix.get_figure() 

    SAVE_PATH =  "paraphrase/figs/cm_train_" + out_file_train +".png"
    figure.savefig(SAVE_PATH)
    plt.close(figure)

    return clf

# Run the trained model on the test dataset, which the model has not seen
def evaluate_model(clf, fname1, fname2, flabel, out_file_train, out_file_test):

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
    abs_diff = np.abs(test_vectors1 - test_vectors2)
    elem_prod = test_vectors1 * test_vectors2

    combined_test = np.concatenate((test_vectors1, 
                        test_vectors2, abs_diff,elem_prod), axis = 1)
    print(combined_test.shape)  
    t_labels = np.array(test_labels['labels'])
    print(t_labels.shape)    

    print("Metrics for test dataset......")       

    t_preds = clf.predict(combined_test) 
    t_pred_probs = clf.predict_proba(combined_test)

    print("Predictions for 10 are", t_preds[0:10])
    print("Prediction probs for 10 are", t_pred_probs[0:10])

    print("Accuracy for test set is:", clf.score(combined_test, t_labels)) 
    print("F1score for test set is: ", f1_score(t_labels, t_preds,  average=None))

    ct_matrix = confusion_matrix(t_labels, t_preds, labels=[0, 1], normalize = "true")
    print(ct_matrix)

    df_cm = pd.DataFrame(ct_matrix, index = [0, 1] ,columns = [0, 1])
    tmatrix = sns.heatmap(df_cm, annot=True, cmap='Blues')
    #plt.figure()
    figure1 = tmatrix.get_figure() 

    SAVE_PATH =  "paraphrase/figs/cm_train_" + out_file_train + "_test_" + out_file_test + ".png"   
    figure1.savefig(SAVE_PATH)
    plt.close(figure1)

    return clf, combined_test, test_data_1, test_data_2


# Identify the verbs in the sentences, get probability scores for pairs, identify indirect speech 
def generate_scored_file(clf, combined_test, test_data_1, test_data_2, out_file_train, out_file_test):

    ## Get paraphrase pairs with high probability ( >= 95)
    df1 = pd.DataFrame(columns=['sent1','length1', 'indirect words sent1', 'count of verbs sent1', 'verbs in sent1'])
    df2 = pd.DataFrame(columns=['sent2','length2', 'indirect words sent2', 'count of verbs sent2', 'verbs in sent2',
    'prob_score'])
    count1 = 0
    count2 = 0

    # Check for these words in the sentence pair
    indirect_quotes=["said", "added", "according"]

    for item in combined_test:
        #print(item.reshape(1,-1).shape) # Shape (1,1536)
        pred_item = clf.predict_proba(item.reshape(1,-1))
        #print(pred_item.shape) # Shape (1,2)
        
        # Threshold for paraphrase probability
        if pred_item.item((0,1)) >= 0.00:
            # print(item.shape) # Shape (1536,)
            new_item = item # preserve shape (1536,)
            # print(new_item.shape) # Shape (1, 1536)
            new_item = np.hsplit(new_item, 4)
            # print(new_item[0].shape, new_item[1].shape) # Shape (384, )
            
            # Retrieve first sentence
            for num, vector in enumerate(test_data_1['embeddings']):
                # print(vector.shape) # Shape (384, )
                if np.array_equal(vector, new_item[0]):
                    count1+=1
                    #print("yes with {:.2f} prob \t".format(pred_item.item((0,1))), end = '')
                    #print(test_data_1['sentences'][num])
                    # Make a dataset with sentence, length and scores
                    list_of_words1 = test_data_1['sentences'][num].split(" ")

                    # Get list of verbs 
                    verbs = get_verbs(test_data_1['sentences'][num])

                    num_words1 = len(list_of_words1)
                    temp_indirect_list = []
                    quote_count = 0
                    # Check for indirect quotes (if more words, make a list of the words )
                    '''for word in list_of_words1:
                        if word.lower() == "said" or word.lower() == "according" or word.lower() == "added":
                            temp_indirect_list.append(word)
                            quote_count+=1
                    
                    if quote_count==0:
                        temp_indirect_list.append("No")'''

                    # Faster method to do the same thing
                    overap_words = set(indirect_quotes) & set([word.lower().rstrip(".") for word in list_of_words1])
                    if bool(overap_words) == True:
                        temp_indirect_list = list(overap_words)
                    else:
                        temp_indirect_list.append("no")

                    df1.loc[count1] = [test_data_1['sentences'][num]] + [num_words1] + [temp_indirect_list] + [len(verbs)] + [verbs]
                    break
            

            # Retrieve second sentence
            for num, vector in enumerate(test_data_2['embeddings']):
                # print(vector.shape) # Shape (384, )
                if np.array_equal(vector, new_item[1]):
                    count2+=1
                    #print("yes with {:.2f} prob \t".format(pred_item.item((0,1))), end = '')
                    #print(test_data_2['sentences'][num])
                    # Make a dataset with sentence, length and scores
                    list_of_words2 = test_data_2['sentences'][num].split(" ")

                    # Get list of verbs 
                    verbs = get_verbs(test_data_2['sentences'][num])

                    num_words2 = len(list_of_words2)
                    temp_indirect_list = []
                    quote_count = 0
                    # Check for indirect quotes (if more words, make a list of the words )
                    '''for word in list_of_words2:
                        if word.lower() == "said" or word.lower() == "according" or word.lower() == "added":
                            temp_indirect_list.append(word)
                            quote_count+=1
                    
                    if quote_count==0:
                        temp_indirect_list.append("No")'''

                    overap_words = set(indirect_quotes) & set([word.lower().rstrip(".") for word in list_of_words2])
                    if bool(overap_words) == True:
                        temp_indirect_list = list(overap_words)
                    else:
                        temp_indirect_list.append("no")

                    df2.loc[count2] = [test_data_2['sentences'][num]] + [num_words2] + [temp_indirect_list] + [len(verbs)] + [verbs] + [pred_item.item((0,1))]
                    break

        elif pred_item.item((0,1)) <= 0.05:
            # print(item.shape) # Shape (1536,)
            new_item = item # preserve shape (1536,)
            # print(new_item.shape) # Shape (1, 1536)
            new_item = np.hsplit(new_item, 4)
            # print(new_item[0].shape, new_item[1].shape) # Shape (384, )

            # Retrieve first sentence
            for num, vector in enumerate(test_data_1['embeddings']):
                # print(vector.shape) # Shape (384, )
                if np.array_equal(vector, new_item[0]):
                    print("yes with 5% prob \t", end = '')
                    print(test_data_1['sentences'][num])
                    break

            # Retrieve second sentence
            for num, vector in enumerate(test_data_2['embeddings']):
                # print(vector.shape) # Shape (384, )
                if np.array_equal(vector, new_item[1]):
                    print("yes with 5% prob \t", end = '')
                    print(test_data_2['sentences'][num])
                    break

    #print(df1.head())
    #print(df2.head())
    final_df = pd.concat([df1, df2], axis=1)
    print(final_df.head())

    SAVE_PATH = "paraphrase/figs/paraphr_trainset_" + out_file_train + "_testset_" + out_file_test + ".csv" 
    final_df.to_csv(SAVE_PATH)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train", help="train dataset specifier")
    parser.add_argument("-ev", "--eval", help="eval dataset specifier")

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
    classifier = run_model(X_train, X_test, y_train, y_test, args.train)

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

    # Evaluate the model on the test dataset
    test_classifier, combined_vec, s_vec1, s_vec2 = evaluate_model(classifier, eval_fname1, 
                                                                eval_fname2, eval_flabel, args.train, args.eval)

    # Generate the .csv file with the scored sentence pairs 
    generate_scored_file(test_classifier, combined_vec, s_vec1, s_vec2, args.train, args.eval)
      
if __name__ == '__main__':
    main()
