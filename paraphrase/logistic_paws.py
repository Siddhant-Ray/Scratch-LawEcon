import json, pickle
import os, sys, time, math, random
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

'''a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

print(np.concatenate((a, b), axis = 1))
print(np.concatenate((a, b), axis = 1).shape)
exit()'''

# Load full dataset with combined NLI pairs
with open('paraphrase/data/train_embeddings_paws1.pkl', "rb") as em1:
    stored_data_1 = pickle.load(em1)
with open('paraphrase/data/train_embeddings_paws2.pkl', "rb") as em2:
    stored_data_2 = pickle.load(em2)
with open('paraphrase/data/train_labels_paws.pkl', "rb") as lbl:
    stored_labels = pickle.load(lbl)

# Combine vectors u, v as concat(u, v, |u - v|, u * v)
input_vectors1 = stored_data_1['embeddings']
print(type(input_vectors1), input_vectors1.shape) # Shape (9076, 384)
input_vectors2 = stored_data_2['embeddings']
print(type(input_vectors2), input_vectors2.shape) # Shape (9076, 384)

abs_diff_vectors = np.abs(input_vectors1 - input_vectors2)
print(type(abs_diff_vectors), abs_diff_vectors.shape) # Shape (9076, 384)

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


def run_model():

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
    #figure.savefig("paraphrase/figs/cm_train_paws.png")
    plt.close(figure)


    print("Testing on PAWS dataset, trained on PAWS train dataset .......")

    # Load test dataset 
    with open('paraphrase/data/test_embeddings_paws1.pkl', "rb") as _em1:
        test_data_1 = pickle.load(_em1)
    with open('paraphrase/data/test_embeddings_paws2.pkl', "rb") as _em2:
        test_data_2 = pickle.load(_em2)
    with open('paraphrase/data/test_labels_paws.pkl', "rb") as _lbl:
        test_labels = pickle.load(_lbl)

    test_vectors1, test_vectors2 = test_data_1['embeddings'], test_data_2['embeddings']
    abs_diff = np.abs(test_vectors1 - test_vectors2)
    elem_prod = test_vectors1 * test_vectors2

    combined_test = np.concatenate((test_vectors1, 
                        test_vectors2, abs_diff,elem_prod), axis = 1)
    print(combined_test.shape)  
    t_labels = np.array(test_labels['labels'])
    print(t_labels.shape)    

    print("Metrics for only PAWS test dataset......")       

    t_preds = clf.predict(combined_test) 
    t_pred_probs = clf.predict_proba(combined_test)

    print("Predictions for 10 are", t_preds[0:10])
    print("Prediction probs for 10 are", t_pred_probs[0:10])

    print("Accuracy for PAWS test set is:", clf.score(combined_test, t_labels)) 
    print("F1score for PAWS test set is: ", f1_score(t_labels, t_preds,  average=None))

    ct_matrix = confusion_matrix(t_labels, t_preds, labels=[0, 1], normalize = "true")
    print(ct_matrix)

    df_cm = pd.DataFrame(ct_matrix, index = [0, 1] ,columns = [0, 1])
    tmatrix = sns.heatmap(df_cm, annot=True, cmap='Blues')
    #plt.figure()
    figure1 = tmatrix.get_figure()    
    #figure1.savefig("paraphrase/figs/cm_test_paws_train_paws.png")
    plt.close(figure1)


    ## Get paraphrase pairs with high probability ( >= 80)
    df1 = pd.DataFrame(columns=['sent1','length1'])
    df2 = pd.DataFrame(columns=['sent2','length2','prob_score'])
    count1 = 0
    count2 = 0

    for item in combined_test:
        #print(item.reshape(1,-1).shape) # Shape (1,1536)
        pred_item = clf.predict_proba(item.reshape(1,-1))
        #print(pred_item.shape) # Shape (1,2)
        
        # Threshold for paraphrase probability
        if pred_item.item((0,1)) >= 0.70:
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
                    print("yes with 70% prob \t", end = '')
                    print(test_data_1['sentences'][num])
                    # Make a dataset with sentence, length and scores
                    num_words1 = len(test_data_1['sentences'][num].split(" "))
                    df1.loc[count1] = [test_data_1['sentences'][num]] + [num_words1]
                    break
            

            # Retrieve second sentence
            for num, vector in enumerate(test_data_2['embeddings']):
                # print(vector.shape) # Shape (384, )
                if np.array_equal(vector, new_item[1]):
                    count2+=1
                    print("yes with 70% prob \t", end = '')
                    print(test_data_2['sentences'][num])
                    # Make a dataset with sentence, length and scores
                    num_words2 = len(test_data_2['sentences'][num].split(" "))
                    df2.loc[count2] = [test_data_2['sentences'][num]] + [num_words2] + [pred_item.item((0,1))]
                    break

        elif pred_item.item((0,1)) <= 0.10:
            # print(item.shape) # Shape (1536,)
            new_item = item # preserve shape (1536,)
            # print(new_item.shape) # Shape (1, 1536)
            new_item = np.hsplit(new_item, 4)
            # print(new_item[0].shape, new_item[1].shape) # Shape (384, )

            # Retrieve first sentence
            for num, vector in enumerate(test_data_1['embeddings']):
                # print(vector.shape) # Shape (384, )
                if np.array_equal(vector, new_item[0]):
                    print("yes with 10% prob \t", end = '')
                    print(test_data_1['sentences'][num])
                    break

            # Retrieve second sentence
            for num, vector in enumerate(test_data_2['embeddings']):
                # print(vector.shape) # Shape (384, )
                if np.array_equal(vector, new_item[1]):
                    print("yes with 10% prob \t", end = '')
                    print(test_data_2['sentences'][num])
                    break

    #print(df1.head())
    #print(df2.head())
    final_df = pd.concat([df1, df2], axis=1)
    print(final_df.head())
    final_df.to_csv('paraphrase/figs/paraphr_pawstestset.csv')

      
if __name__ == '__main__':
    run_model()
