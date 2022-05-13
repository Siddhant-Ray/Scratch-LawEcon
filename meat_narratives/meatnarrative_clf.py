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

from sklearn.model_selection import train_test_split

from transformers import AutoModel, AutoTokenizer
from transformers import BertTokenizer, BertForSequenceClassification
## Global path and dicts
DIR = "meat_narratives/data/"
BERT_MODEL = "bert-base-uncased"
BERT_MODEL_GERMAN = "bert-base-german-cased"

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
    '''
        A utility class to clean the data and make the labels consistent.
    '''

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
        dataframe['statment_reference'] = dataframe.apply(lambda row : statement_reference_dict[row['statement_reference']], axis = 1)

        return dataframe

    @staticmethod
    def create_features_and_labels(dataframe):
        features = dataframe["text"]
        multilabel_dict = {
            "type": dataframe['statement_type'],
            "topic": dataframe['statement_topic'],
            "valence": dataframe['topic_valence'],
            "reference": dataframe['statment_reference']
        }

        return features, multilabel_dict

class MeatDataset(Dataset):
    '''
        PyTorch dataset class helper
    '''
    def __init__(self,features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        assert len(self.labels["type"]) == len(self.labels["topic"])
        assert len(self.labels["valence"]) == len(self.labels["reference"])
        assert len(self.labels["type"]) == len(self.labels["reference"])
        return len(self.labels["type"])

    def __getitem__(self, idx):
        single_feature = {key: (val[idx]) for key, val in self.features.items()}

        label0 = self.labels["type"].to_list()[idx]
        label1 = self.labels["topic"].to_list()[idx]
        label2 = self.labels["valence"].to_list()[idx]
        label3 = self.labels["reference"].to_list()[idx]

        sample = {'feature':single_feature, 'labels': {'label_type':label0, 'label_topic':label1, 
                                                        'label_valence':label2, 'label_reference':label3 }}
        
        return sample   

class Identity(nn.Module):
    '''
        Class to rewrite the fully connected layer after the BERT layers
    '''
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class MultilabelClassifier(nn.Module):
    '''
        Actual classification network
    '''
    def __init__(self, n_type, n_topic, n_valence, n_reference):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(BERT_MODEL_GERMAN, num_labels = 10)
        self.model.dropout = Identity()
        self.model.classifier = Identity()

        self.type = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=768, out_features=n_type)
        )
        self.topic = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=768, out_features=n_topic)
        )
        self.valence = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=768, out_features=n_valence)
        )
        self.reference = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=768, out_features=n_reference)
        )

    def forward(self, x):
        x = self.model(x)
        
        return {
            'type': self.topic(x),
            'topic': self.type(x),
            'valence': self.valence(x),
            'reference': self.reference(x)
        }

# Combined loss function for all output labels
def criterion(loss_func,outputs,samples,device):
   losses = 0
   for i, key in enumerate(outputs):
       losses += loss_func(outputs[key], samples['labels'][f'label_{key}'].to(device))
   return losses

def run(args):

    # Load the cleaned dataset
    df = DataCleaner.generate_dataset(DIR)
    processed_df = DataCleaner.pre_process_data_to_numeric_labels(df)
    
    # Create lists of sentences and labels
    features, labels = DataCleaner.create_features_and_labels(processed_df)
    print(len(features)); print([len(labels[key]) for key in labels.keys()])
    list_of_sentences = features.to_list()
    print(list_of_sentences[0:10])
    print(len(list_of_sentences))

    # Tokenize sentence inputs
    tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased")
    tokenized_inputs = tokenizer(list_of_sentences, return_tensors='pt', truncation=True, padding=True)
    print(tokenized_inputs.items())
    print(len(tokenized_inputs))

    # Prepare dataset items
    text_data = MeatDataset(tokenized_inputs, labels)
    
    train_len = int(text_data.__len__()*0.8)
    val_len = int(text_data.__len__()*0.2)
    print(f'train len is : {train_len}', f'val len is {val_len}')
    train_set, val_set = torch.utils.data.random_split(text_data , [train_len, val_len])

    # Prepare data loaders
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, 
                                num_workers=0, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, 
                                num_workers=0, drop_last=False)
    
    # Get one sample batch from the dataloader
    sample = next(iter(train_loader))
    print("Keys in our sample batch: {}".format(sample.keys()))
    print("Features in our sample batch: {}".format(sample['feature']))
    print("Size for the target in our sample batch: {}".format(len(sample['labels']['label_type'])))
    print("Targets for each batch in our sample: {}".format(sample['labels']['label_type']))

    # Initialise classifier
    classifier = MultilabelClassifier(3,4,2,17)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    classifier.to(device)  

    


def main():
    test_args = ""
    run(test_args)

if __name__=="__main__":
    main()
            
