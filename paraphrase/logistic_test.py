import json, pickle
from operator import le
import os, sys, time, math, random
import argparse

import numpy as np
import pandas as pd

PATH = "paraphrase/figs/"

def read_csv(path):
    df = pd.read_csv(path)
    return df

def filter_dataframe(df, threshold):
    threshold = float(threshold)
    df = df.loc[df['prob_score'] >= threshold]
    df = df.drop(columns=['indirect words sent1','count of verbs sent1','verbs in sent1'])
    df = df.drop(columns=['indirect words sent2','count of verbs sent2','verbs in sent2'])
    df = df.rename(columns={'prob_score': 'paraphrase_probability'})
    return df 

def save_filtered_csv(df, save_path, train_set, threshold):
    save_path = PATH + train_set + "_trainset_" + "_filtered_paraprob_greater_than" + str(threshold) + ".csv"
    df.to_csv(save_path, index = False)


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


if __name__== '__main__':
    main()

