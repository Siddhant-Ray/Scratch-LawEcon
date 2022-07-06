import os, json, pickle, sys, time
from re import M
import argparse
import pandas as pd
import numpy as np

PATH = "simplification_clustering/datasets/"

# Load mapping data
def load_data_manf_map(path):
    non_simplified_sentences = []
    simplified_sentences = []

    mapping_df = pd.DataFrame(columns=['original', 'simplified'])

    data = open(path+"manifesto_simplified.txt", 'r').readlines()
    for line in data:
        line_val = json.loads(line)
        for idx, item in enumerate(line_val['simplified']):
            if len(item["text"].split()) >=6:   
                simplified_sentences.append(item['text'])
                non_simplified_sentences.append(line_val['original'])

    mapping_df['original'] = non_simplified_sentences
    mapping_df['simplified'] = simplified_sentences
    
    return mapping_df

# Load actual labels
def load_actual_labels(path):

    data = pd.read_csv(path+"manifesto_map.txt", sep='\t', on_bad_lines='skip')
    data.columns = ['original', 'label1', 'label2', 'original2']

    return data

# Map between data frames
def map_df(mapping_df, actual_labels):
    
    actual_labels.set_index('original',inplace=True)
    label1_dict = actual_labels.to_dict()['label1']
    label2_dict = actual_labels.to_dict()['label2']    
    
    for key in label1_dict.keys():
        mapping_df.loc[mapping_df['original'] == key, 'label1'] = label1_dict[key]
        mapping_df.loc[mapping_df['original'] == key, 'label2'] = label2_dict[key]

    return mapping_df

def run(args):
    path = PATH + args.path + "/"
    mapping_df = load_data_manf_map(path)
    actual_labels = load_actual_labels(path)
    print(mapping_df.head())
    mapping_df_with_labels = map_df(mapping_df, actual_labels)

    mapping_df_with_labels.to_csv(path+"mapping_data.csv", index=False)
   
# Main
def main():
    parser = argparse.ArgumentParser(description='Evaluate dataset')
    parser.add_argument('--path', type=str, default=PATH, help='Path to the dataset')
    args = parser.parse_args()

    run(args)

if __name__ == '__main__':
    main()