from cgi import test
from dataclasses import dataclass
from unittest import TestCase
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import transformers
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
from torchmetrics.functional import accuracy

@dataclass
class Config:
    file_path = 'meat_narratives/edgelist_organisations_concepts.csv'
    lr = 1e-5
    max_len = 128
    train_bs = 64
    valid_bs = 32
    train_pcent = 0.99
    num_workers = 0
    bert_model = 'bert-base-german-cased'
    tokenizer = transformers.AutoTokenizer.from_pretrained(bert_model, do_lower_case=True)

class BertData(Dataset):
    def __init__(self, sentences, targets):
        self.sentences = sentences
        self.targets = targets
        self.tokenizer = Config.tokenizer
        self.max_len = Config.max_len

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens = True,
            max_length = Config.max_len,
            pad_to_max_length=True,
            truncation='longest_first'
        )

        ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
        token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long)
        targets = torch.tensor(self.targets[idx], dtype=torch.float)

        return {'ids': ids,
                'mask': mask,
                'token_type_ids': token_type_ids,
                'targets': targets
                }

class BERTModel(pl.LightningModule):
    def __init__(self) -> None:
        super(BERTModel, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(Config.bert_model)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)
        self.all_targets = []
        self.train_loss_fn = nn.BCEWithLogitsLoss()
        self.valid_loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(self, ids, mask, token_type_ids) -> torch.Tensor:
        _, output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output = self.drop(output)
        output = self.out(output)
        return output
    
    def prepare_data(self) -> None:
        # Load the data, encode, shuffle and split it
        data = pd.read_csv("meat_narratives/edgelist_organisations_concepts.csv", on_bad_lines='skip', delimiter=";")
        data["text"] = data.apply(lambda row: row["text"].replace("\n",""), axis=1)
        data = data[['text', 'agreement']]
        
        nb_training_samples = int(Config.train_pcent * len(data))

        self.train_data = data[:nb_training_samples]
        self.valid_data = data[nb_training_samples:]

        # Make Training and Validation Datasets
        self.training_set = BertData(
            sentences=self.train_data['text'].values,
            targets=self.train_data['agreement'].values
        )

        self.validation_set = BertData(
           sentences=self.valid_data['text'].values,
            targets=self.valid_data['agreement'].values
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.training_set,
            batch_size=Config.train_bs,
            shuffle=False,
            num_workers=Config.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.validation_set,
            batch_size=Config.valid_bs,
            shuffle=False,
            num_workers=Config.num_workers,
        )
        return val_loader
    
    def training_step(self, batch, batch_idx):
        ids = batch['ids'].long()
        mask = batch['mask'].long()
        token_type_ids = batch['token_type_ids'].long()
        targets = batch['targets'].float()

        outputs = self(ids=ids, mask=mask, token_type_ids=token_type_ids)

        train_loss = self.train_loss_fn(outputs, targets.view(-1, 1))
        train_acc = ((outputs > 0) == targets).float().mean().item()

        return {'loss': train_loss, 'acc':train_acc}
    
    def validation_step(self, batch, batch_idx):
        ids = batch['ids'].long()
        mask = batch['mask'].long()
        token_type_ids = batch['token_type_ids'].long()
        targets = batch['targets'].float()

        outputs = self(ids=ids, mask=mask, token_type_ids=token_type_ids)
        
        self.all_targets.extend(targets.cpu().detach().numpy().tolist())
    
        valid_loss = self.valid_loss_fn(outputs, targets.view(-1, 1))
        valid_acc = ((outputs > 0) == targets).float().mean().item()
        return {'loss': valid_loss, 'acc':valid_acc}
    
    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return transformers.AdamW(optimizer_parameters, lr=Config.lr)

    def training_epoch_end(self,outputs):
        print(outputs)
        exit()
        loss_tensor_list = [item['loss'].to('cpu').numpy() for item in outputs['loss']]
        acc_tensor_list = [item for item in outputs['acc']]

        print('Avg train loss per epoch', np.mean(np.array(loss_tensor_list)), on_step=False, on_epoch=True)
        print('Avg train acc per epoch', np.mean(np.array(acc_tensor_list)), on_step=False, on_epoch=True)

    def val_epoch_end(self,outputs):
        loss_tensor_list = [item['loss'].to('cpu').numpy() for item in outputs['loss']]
        acc_tensor_list = [item for item in outputs['acc']]

        print('Avg val loss per epoch', np.mean(np.array(loss_tensor_list)), on_step=False, on_epoch=True)
        print('Avg val acc per epoch', np.mean(np.array(acc_tensor_list)), on_step=False, on_epoch=True)



model = BERTModel()
trainer = pl.Trainer(max_epochs=10, gpus=1,log_every_n_steps=10)
trainer.fit(model)





