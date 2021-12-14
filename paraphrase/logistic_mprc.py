import json, pickle
import os, sys, time, math, random

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

# Load only mprc dataset 
with open('paraphrase/data/mprc_embeddings_1.pkl', "rb") as _em1:
    stored_data_mprc_1 = pickle.load(_em1)
with open('paraphrase/data/mprc_embeddings_2.pkl', "rb") as _em2:
    stored_data_mprc_2 = pickle.load(_em2)
with open('paraphrase/data/mprc_labels.pkl', "rb") as _lbl:
    stored_labels_mprc = pickle.load(_lbl)


input_vectors1 = stored_data_mprc_1['embeddings']
print(type(input_vectors1), input_vectors1.shape)
input_vectors2 = stored_data_mprc_2['embeddings']
print(type(input_vectors2), input_vectors2.shape)

abs_diff_vectors = np.abs(input_vectors1 - input_vectors2)
print(type(abs_diff_vectors), abs_diff_vectors.shape)

# product_of_vectors = np.einsum('ij,ij->i', input_vectors1, input_vectors2)[..., None]
product_of_vectors = input_vectors1 * input_vectors2
print(type(product_of_vectors), product_of_vectors.shape)

input_combined_vectors1 = np.concatenate((input_vectors1, 
                        input_vectors2, abs_diff_vectors,product_of_vectors), axis = 1)

input_combined_vectors2 = np.concatenate((input_vectors2, 
                        input_vectors1, abs_diff_vectors,product_of_vectors), axis = 1)

input_combined_vectors_all = np.concatenate((input_combined_vectors1, input_combined_vectors2), axis = 0)                        


print(type(input_combined_vectors_all), input_combined_vectors_all.shape)
labels = np.array(stored_labels_mprc['labels'])
labels_all = np.concatenate([labels] * 2, axis=0)
print(type(labels_all), labels_all.shape)


X_train, X_test, y_train, y_test = train_test_split(input_combined_vectors_all, labels_all, 
                                    test_size=0.2, shuffle = True, random_state = 0)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

'''for i in range(10):
    print("X_train", X_train.item(i))
    print("y_train", y_train.item(i))'''


def run_model():

    clf = LogisticRegression(penalty='l2',random_state=0, max_iter=1000,
                        n_jobs = -1).fit(X_train, y_train)

    preds = clf.predict(X_test)
    pred_probs = clf.predict_proba(X_test)

    print("Predictions for 100 are", preds[0:100])
    print("Prediction probs for 100 are", pred_probs[0:100])

    print("Accuracy is:", clf.score(X_test, y_test))
    print("F1score is: ", f1_score(y_test, preds,  average=None))

    # Form and print confusion matrix, plot heatmap
    c_matrix = confusion_matrix(y_test, preds, labels=[0, 1], normalize = "true")
    print(c_matrix)

    df_cm = pd.DataFrame(c_matrix, index = [0, 1] ,columns = [0, 1])
    matrix = sns.heatmap(df_cm, annot=True, cmap='Blues')
    plt.figure()
    figure = matrix.get_figure()    
    figure.savefig("paraphrase/figs/cm_mprc.png")


    print("Testing on MPRC dataset, trained on MPRC pairs.......")

    # Load test dataset 
    with open('paraphrase/data/test_embeddings_1.pkl', "rb") as _em1:
        test_data_1 = pickle.load(_em1)
    with open('paraphrase/data/test_embeddings_2.pkl', "rb") as _em2:
        test_data_2 = pickle.load(_em2)
    with open('paraphrase/data/test_labels.pkl', "rb") as _lbl:
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
    t_pred_probs = clf.predict_proba(X_test)

    print("Predictions for 10 are", t_preds[0:10])
    print("Prediction probs for 10 are", t_pred_probs[0:10])

    print("Accuracy for test set is:", clf.score(combined_test, t_labels)) 
    print("F1score for test set is: ", f1_score(t_labels, t_preds,  average=None))

    ct_matrix = confusion_matrix(t_labels, t_preds, labels=[0, 1], normalize = "true")
    print(c_matrix)

    df_cm = pd.DataFrame(ct_matrix, index = [0, 1] ,columns = [0, 1])
    tmatrix = sns.heatmap(df_cm, annot=True, cmap='Blues')
    plt.figure()
    figure = tmatrix.get_figure()    
    figure.savefig("paraphrase/figs/cm_test_mprc.png")


if __name__ == '__main__':
    run_model()
