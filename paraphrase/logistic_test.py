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
    df.loc[df['column_name'] >= threshold]
    return df 

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

    print(full_path)

