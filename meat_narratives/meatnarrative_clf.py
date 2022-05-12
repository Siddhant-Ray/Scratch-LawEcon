import random, os
import numpy as np 
import pandas as pd

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import scipy.io
import pandas as pd
import seaborn as sns

## Global path and dicts
DIR = "meat_narratives/data/"

statement_type_dict = { "narrative": 0,
                        "goal": 1,
                        "instrument": 2
                    }

statement_topic_dict = { "meat": 0,
                        "substitute": 1,
                        "plant based": 2,
                        "all": 3
                    }

topic_valence_dict = { "pro": 0,
                        "contra": 1,
                    }

statement_reference_dict = {
            "health" : 0,
            "environment" : 1,
            "climate" : 2,
            "biodiversity" : 3,
            "land usage" : 4, 
            "water usage and quality" : 5,
            "deforestation" : 6,
            "animal welfare" : 7,
            "working conditions" : 8,
            "pandemics and epizootic diseases" : 9,
            "antibiotics" : 10,
            "economy" : 11,
            "moral and ethic" : 12,
            "taste and texture" : 13,
            "world food supply" : 14,
            "highly processed" : 15,
            "social fairness" : 16 
}    

# Cleaner class
class DataCleaner():

    def __init__(self, fil_dir, statement_type_dict, statement_topic_dict,
                topic_valence_dict, statement_reference_dict):
        self.DIR = fil_dir
        self.statement_type_dict = statement_type_dict
        self.statement_topic_dict = statement_topic_dict
        self.topic_valence_dict = topic_valence_dict
        self.statement_reference_dict = statement_reference_dict

    @staticmethod
    def generate_dataset(fil_dir):
        cols = ['text','statement','statement_type','statement_topic',
                                'topic_valence', 'statement_reference']
        final_df = pd.DataFrame(columns = cols )
        for file in os.listdir(fil_dir):
            if file.endswith(".csv"):
                df = pd.read_csv(fil_dir+file, delimiter=";")
                df = df[cols]
                df.dropna(inplace=True)
                df.reset_index(drop=True, inplace=True)
                final_df = pd.concat([final_df, df])

        statements = final_df['statement_type'].to_list()
        statements = ['instrument' if item.casefold() == 'policy instrument' else item for item in statements]
        statements = ['goal' if item.casefold() == 'policy goal' else item for item in statements]

        valence = final_df['topic_valence'].to_list()
        valence = ['contra' if item.casefold() == 'contra meat' else item for item in valence]
        valence = ['contra' if item.casefold() == 'contra plant-based' else item for item in valence]
        valence = ['pro' if item.casefold() == 'pro plant-based' else item for item in valence]
        valence = ['pro' if item.casefold() == 'pro meat' else item for item in valence]
        valence = [item.strip(" ") for item in valence]

        final_df['statement_type'] = statements
        final_df['topic_valence'] = valence
        
        final_df['statement_topic'] = final_df['statement_topic'].str.replace("-"," ")
        references = final_df['statement_reference'].tolist()
        references = [item.rstrip(" ") for item in references]
        references = ['economy' if item.casefold() == 'economic' else item for item in references]
        references = ['animal welfare' if item.casefold() == 'ann' else item for item in references]
        references = ['moral and ethic' if item.casefold() == 'moral and ethics' else item for item in references]
        references = ['water usage and quality' if item.casefold() == 'water-usage and quality' else item for item in references]

        final_df['statement_reference'] = references

        label_cols = ['statement','statement_type','statement_topic',
                                'topic_valence', 'statement_reference']

        final_df[label_cols] = final_df.apply(lambda row : row[label_cols].str.lower(), axis = 1)

        return final_df

    @staticmethod
    def replace_key_value(list,type_dict):
        for idx in range(len(list)-1):
            list[idx] = type_dict[list[idx]]
        return list

    @staticmethod
    def pre_process_data_to_numeric_labels(dataframe):
        
        dataframe['statement_type'] = dataframe.apply(lambda row : statement_type_dict[row['statement_type']], axis = 1)
        dataframe['statement_topic'] = dataframe.apply(lambda row : statement_topic_dict[row['statement_topic']], axis = 1)
        dataframe['topic_valence'] = dataframe.apply(lambda row : topic_valence_dict[row['topic_valence']], axis = 1)
        dataframe['topic_valence'] = dataframe.apply(lambda row : statement_reference_dict[row['statement_reference']], axis = 1)

        return dataframe

    @staticmethod
    def create_features_and_labels(dataframe):
        features = dataframe["text"]
        multilabel_dict = {
            "type": dataframe['statement_type'],
            "topic": dataframe['statement_topic'],
            "valence": dataframe['topic_valence'],
            "reference": dataframe['topic_valence']
        }

        return features, multilabel_dict

def run(args):
    df = DataCleaner.generate_dataset(DIR)
    processed_df = DataCleaner.pre_process_data_to_numeric_labels(df)
    
    features, labels = DataCleaner.create_features_and_labels(processed_df)
    print(len(features)); print([len(labels[key]) for key in labels.keys()])


def main():
    test_args = ""
    run(test_args)

if __name__=="__main__":
    main()
            
