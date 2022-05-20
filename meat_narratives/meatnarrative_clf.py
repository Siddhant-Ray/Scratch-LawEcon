from calendar import c
import random, os
import numpy as np 
import pandas as pd

from collections import Counter
import argparse

import json 

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
from sklearn.metrics import f1_score

from transformers import AutoModel, AutoTokenizer
from transformers import BertTokenizer, BertForSequenceClassification

from tqdm import tqdm

# Set seeds 
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

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

class color:
   RED = '\033[91m'
   BOLD = '\033[1m'
   END = '\033[0m'

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
    def count_label_occurrences(dataframe):
        counter_1 = Counter(dataframe['statement_type'])
        counter_2 = Counter(dataframe['statement_topic'])
        counter_3 = Counter(dataframe['topic_valence'])
        counter_4 = Counter(dataframe['statement_reference'])

        return counter_1, counter_2, counter_3, counter_4

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

        self.s_type = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=768, out_features=n_type)
        )
        self.s_topic = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=768, out_features=n_topic)
        )
        self.s_valence = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=768, out_features=n_valence)
        )
        self.s_reference = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=768, out_features=n_reference)
        )

    def forward(self, x, attn):
        out = self.model(x, attn)
        
        return {
            'type': self.s_type(out.logits),
            'topic': self.s_topic(out.logits),
            'valence': self.s_valence(out.logits),
            'reference': self.s_reference(out.logits)
        }

# Combined loss function for all output labels
def criterion(loss_func,outputs,samples,device):
   losses = 0
   for i, key in enumerate(outputs):
       losses += loss_func(outputs[key], samples['labels'][f'label_{key}'].to(device))
   return losses

# Train function
def train_model(model,device,lr_rate,epochs,train_loader):

    num_epochs = epochs
    losses = []
    checkpoint_losses = []

    # Freeze the BERT layers
    for param in model.model.bert.parameters():
        param.requires_grad = False

    # Specify optimiser only for FC layers
    optimizer_params = [
                {'params': model.s_type.parameters()},
                {'params': model.s_topic.parameters()},
                {'params': model.s_valence.parameters()},
                {'params': model.s_reference.parameters()}
            ]
    
    optimizer = torch.optim.Adam(optimizer_params, lr=lr_rate)
    n_total_steps = len(train_loader)

    loss_func = nn.CrossEntropyLoss()

    for epoch in tqdm(range(num_epochs)):
        model.train()

        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch['feature']['input_ids'].to(device)
            attention_mask = batch['feature']['attention_mask'].to(device)
            
            target = batch

            outputs = model(input_ids, attention_mask)

            loss = criterion(loss_func,outputs, target, device)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if (idx+1) % (int(n_total_steps/1)) == 0:
                checkpoint_loss = torch.tensor(losses).mean().item()
                checkpoint_losses.append(checkpoint_loss)
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{idx+1}/{n_total_steps}], Loss: {checkpoint_loss:.4f}')

    return checkpoint_losses

def validate_model(model,device, val_loader, batch_size):
    model.eval()

    class_list = ["type", "topic", "valence", "reference"]
    actual_targets = {}
    predicted_targets = {}

    for value in class_list:
        actual_targets[value] = []
        predicted_targets[value] = []

    with torch.no_grad():
        val_losses = []
        accuracy_dict = {}
        accuracy_dict["type"] = 0
        accuracy_dict["topic"] = 0
        accuracy_dict["valence"] = 0
        accuracy_dict["reference"] = 0

        for idx, batch in enumerate(val_loader):

            input_ids = batch['feature']['input_ids'].to(device)
            attention_mask = batch['feature']['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)

            correct_dict = {}

            target = batch

            loss_func = nn.CrossEntropyLoss()
            loss = criterion(loss_func,outputs, target, device)
            val_losses.append(loss.item())

            correct_dict["type"] = batch['labels']['label_type'].to(device)
            correct_dict["topic"] = batch['labels']['label_topic'].to(device)
            correct_dict["valence"] = batch['labels']['label_valence'].to(device)
            correct_dict["reference"] = batch['labels']['label_reference'].to(device)
            
            for i,out in enumerate(outputs):
                _, predicted = torch.max(outputs[out],1)
                accuracy = (outputs[out].argmax(-1) == correct_dict[out]).float().mean()
                append_acc = (accuracy.cpu().item())
                accuracy_dict[out] += append_acc 

                pred_outs = predicted.cpu().detach().numpy().tolist()
                true_outs = correct_dict[out].cpu().detach().numpy().tolist()
               
                predicted_targets[out].extend(pred_outs)
                actual_targets[out].extend(true_outs)

        print()
        print("Validation loss is ", torch.tensor(val_losses).mean().item())
        print()
        print("Accuracy of statement type is ", accuracy_dict["type"]/len(val_loader))
        print("Accuracy of statement topic is ", accuracy_dict["topic"]/len(val_loader))
        print("Accuracy of topic valence is ", accuracy_dict["valence"]/len(val_loader))
        print("Accuracy of statement reference is ", accuracy_dict["reference"]/len(val_loader))
    
    return val_losses,accuracy_dict, actual_targets, predicted_targets

def calculate_f1_score(true, pred):
    return f1_score(true, pred, average="macro")

def run(args):

    # Load the cleaned dataset
    df = DataCleaner.generate_dataset(DIR)

    print()
    ## Print counts of all labels
    counter1, counter2, counter3, counter4 = DataCleaner.count_label_occurrences(df)
    print(json.dumps(counter1, indent=2))
    print(json.dumps(counter2, indent=2))
    print(json.dumps(counter3, indent=2))
    print(json.dumps(counter4, indent=2))

    # Create numeric labels
    processed_df = DataCleaner.pre_process_data_to_numeric_labels(df)

    # Create lists of sentences and labels
    features, labels = DataCleaner.create_features_and_labels(processed_df)
    print(len(features)); print([len(labels[key]) for key in labels.keys()])
    list_of_sentences = features.to_list()
    print(list_of_sentences[0:2])
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
    train_loader = DataLoader(train_set, batch_size = args.batch_size, shuffle=True, 
                                num_workers=0, drop_last=False)
    val_loader = DataLoader(val_set, batch_size = args.batch_size, shuffle=False, 
                                num_workers=0, drop_last=False)
    
    # Get one sample batch from the dataloader
    sample = next(iter(train_loader))
    print("Keys in our sample batch: {}".format(sample.keys()))
    print("Features in our sample batch: {}".format(sample['feature']))
    print("Size for the target in our sample batch: {}".format(len(sample['labels']['label_type'])))
    print("Targets for each batch in our sample: {}".format(sample['labels']['label_type']))

    # Initialise classifier
    num_types = len(statement_type_dict.keys())
    num_topics = len(statement_topic_dict.keys())
    num_valence = len(topic_valence_dict.keys())
    num_refs = len(statement_reference_dict.keys())
    print(num_types, num_topics, num_valence, num_refs)

    classifier = MultilabelClassifier(num_types, num_topics,
                                    num_valence, num_refs)
    # print(classifier)
    
    # Choose device to run on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    classifier.to(device)  

    checkpoint_losses = train_model(classifier, device, args.learning_rate,
                                    args.num_epochs, train_loader)

    
    val_losses, accuracy_values, actual_values, predicted_values = validate_model(classifier, device,
                                    val_loader, args.batch_size)

    print()
    for key in actual_values.keys():
        if key != "valence":
            print("F1 macro score on statement {} is ".format(key), calculate_f1_score(actual_values[key], predicted_values[key]))
        else:
            print("F1 macro score on topic {} is ".format(key), calculate_f1_score(actual_values[key], predicted_values[key]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size",
                        type=int, default=64,
                        required=False,
                        help = "select the batch_size")
    parser.add_argument("-nepochs", "--num_epochs",
                        type=int, default=10,
                        required=False,
                        help= "choose number of epochs to run")
    parser.add_argument("-lr", "--learning_rate", 
                        type=float, default=1e-4,
                        required=False,
                        help= "choose learning rate")
    
    args = parser.parse_args()
    print(args)

    run(args)

if __name__=="__main__":
    main()
            
