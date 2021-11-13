import json, pickle 
import random
import math

import torch, torch.nn as nn
torch.manual_seed(0)

import numpy as np
np.random.seed(0)

import random
random.seed(0)

from sentence_transformers import SentenceTransformer

file_path_1 = "paraphrase/MSRParaphraseCorpus/msr_paraphrase_train.txt"
file_path_2 = "paraphrase/multinli_1.0/multinli_1.0_train.jsonl"
file1 = open(file_path_1)
lines_of_sentences = file1.readlines()

# list to store values
labeled_pairs_list = []

for value in lines_of_sentences:
    # split on "\t"
    data = value.split("\t")
    label = data[0] 

    #list of pairs of sentences needed  
    sent_pairs = [data[3].strip("\n"), data[4].strip("\n")]
    #labeled pairs 
    labeled_pairs_list.append([label, sent_pairs])

# remove item 1 with headings
labeled_pairs_list.pop(0)
count_1 = 0
count_0 = 0 

for item in labeled_pairs_list:
    if int(item[0])== 1:
        count_1 += 1 
    else:
        count_0 += 1

#print(count_1,count_0)

with open(file_path_2, 'r') as json_file:
    json_list = list(json_file)

nli_contr_count = 0 
nli_pairs_list = []
for json_str in json_list:
    result = json.loads(json_str)
    if result['annotator_labels']==['contradiction']:
        label = str(0)
        sent1 = result['sentence1']
        sent2 = result['sentence2']
        sent_pairs = [sent1, sent2]
        nli_contr_count+=1
        nli_pairs_list.append([label, sent_pairs])

subset = random.sample(nli_pairs_list, 10000)
#print(subset[0:5])
#print(len(subset))

final_list_of_pairs = random.sample(labeled_pairs_list + subset, len(labeled_pairs_list + subset))
print(len(final_list_of_pairs))
print(final_list_of_pairs[0:5])

#print(len(nli_pairs_list))
#print(nli_pairs_list[0:5])

model = SentenceTransformer('all-MiniLM-L6-v2', device = 'cuda')    
train_labels = [int(item[0]) for item in final_list_of_pairs] 
sentence_1_list = [item[1][0] for item in final_list_of_pairs]
sentence_2_list = [item[1][1] for item in final_list_of_pairs]

#print(len(train_labels),len(sentence_1_list),len(sentence_2_list))
'''embeddings_1 = model.encode(sentence_1_list)
embeddings_2 = model.encode(sentence_2_list)

with open('paraphrase/embeddings_1.pkl', "wb") as fOut1:
    pickle.dump({'sentences': sentence_1_list, 'embeddings': embeddings_1}, fOut1, protocol=pickle.HIGHEST_PROTOCOL)

with open('paraphrase/embeddings_2.pkl', "wb") as fOut2:
    pickle.dump({'sentences': sentence_2_list, 'embeddings': embeddings_2}, fOut2, protocol=pickle.HIGHEST_PROTOCOL)'''

with open('paraphrase/embeddings_1.pkl', "rb") as em1:
    stored_data_1 = pickle.load(em1)

with open('paraphrase/embeddings_2.pkl', "rb") as em2:
    stored_data_2 = pickle.load(em2)

print(len(stored_data_1['embeddings']), type(stored_data_1['embeddings']))
print(len(stored_data_2['embeddings']), type(stored_data_2['embeddings']))


class SNNLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.normal_(self.fc.weight, std=math.sqrt(1 / self.fc.weight.shape[1]))
        nn.init.zeros_(self.fc.bias)

class SimilarityNN(nn.Module):
    """ Simple NN architecture """
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.transform = nn.Sequential(
            nn.AlphaDropout(0.2),
            SNNLinear(input_size, hidden_size),
            nn.SELU(),
            nn.AlphaDropout(0.2),
            SNNLinear(hidden_size, hidden_size // 2),
        )
        self.combination = nn.Sequential(
            nn.SELU(),
            nn.AlphaDropout(0.2),
            SNNLinear(hidden_size // 2, output_size),
        )
        
    def forward(self, input1, input2):
        c1 = self.transform(input1)
        c2 = self.transform(input2)
        return self.combination(c1 + c2)

class DatasetManager(torch.utils.data.Dataset):
    def __init__(self, list_of_sent1, list_of_sent2, class_labels):
        self.list_of_sent1 = list_of_sent1
        self.list_of_sent2 = list_of_sent2
        self.class_labels = class_labels

    # get one sample
    def __getitem__(self, idx):
        
        input_tensor1 = torch.from_numpy(self.list_of_sent1[idx]).float()
        input_tensor2 = torch.from_numpy(self.list_of_sent2[idx]).float()
        target_tensor = torch.tensor(self.class_labels[idx])
    
        return input_tensor1, input_tensor2, target_tensor

    def __len__(self):
        return len(self.list_of_sent1)
    
dataset = DatasetManager(stored_data_1['embeddings'], stored_data_2['embeddings'], train_labels)

_input1, _input2,  _target = dataset.__getitem__(0)
print(_input1.shape, _input2.shape, _target.shape)


val_size = 0.2
val_amount = int(dataset.__len__() * val_size)

train_set, val_set = torch.utils.data.random_split(dataset, [
            (dataset.__len__() - (val_amount)),
            val_amount
])

train_dataloader = torch.utils.data.DataLoader(
            train_set,
            batch_size=64,
            shuffle=True,
)
val_dataloader = torch.utils.data.DataLoader(
            val_set,
            batch_size=64,
            shuffle=False,
)

