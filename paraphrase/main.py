import json, pickle
import os, sys, time, math

import torch
import torch.nn as nn
import numpy as np

from utils import DatasetManager
from utils import train, evaluate
from utils import asMinutes, timeSince

from model import SNNLinear
from model import SimilarityNN

import torch, torch.nn as nn
torch.manual_seed(0)

import numpy as np
np.random.seed(0)

import random
random.seed(0)

from torch.optim.lr_scheduler import *
from tqdm.notebook import tqdm

with open('paraphrase/data/embeddings_1.pkl', "rb") as em1:
    stored_data_1 = pickle.load(em1)

with open('paraphrase/data/embeddings_2.pkl', "rb") as em2:
    stored_data_2 = pickle.load(em2)

with open('paraphrase/data/labels.pkl', "rb") as lbl:
    stored_labels = pickle.load(lbl)

#print(len(stored_data_1['embeddings']), type(stored_data_1['embeddings']))
#print(len(stored_data_2['embeddings']), type(stored_data_2['embeddings']))
#print(len(stored_labels['labels']), type(stored_labels['labels']))

dataset = DatasetManager(stored_data_1['embeddings'], stored_data_2['embeddings'], stored_labels['labels'])

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

## Hyperparameters
learning_rate = 1e-4
hidden_size = 128
input_size = stored_data_1['embeddings'].shape
output_size = 1
epochs = 100 

def trainIters(model, n_iters, embedded, val_embedded, print_every, learning_rate=learning_rate):
    start = time.time()
    plot_losses_train = []
    plot_losses_val =[]
    print_loss_total_train = 0  # Reset every print_every
    plot_loss_total_train = 0  # Reset every plot_every
    
    print_loss_total_val = 0  # Reset every print_every
    plot_loss_total_val = 0  # Reset every plot_every
    
    print_acc_total_train = 0
    plot_acc_total_train = 0
    plot_acc_train = []
    
    print_acc_total_val = 0
    plot_acc_total_val = 0
    plot_acc_val = []
    
    train_epochs = []
    val_epochs = []

    #TODO: Try ADAM
    model_optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-5)
    
    #TODO: Learning rate scheduler
    scheduler = StepLR(model_optimizer, step_size=50, gamma=0.1)

    criterion = nn.BinaryCrossEntropyLoss() 
    
    total_steps = n_iters*len(embedded)
    
    for epoch in range(n_iters):
                
        for local_step, (_input1, _input2, _target) in enumerate(embedded, 1):

            input_tensor1 = _input.to(device)
            input_tensor2 = _input.to(device)

            #noise = torch.randn_like(input_tensor1) * 1e-3 
            #input_tensor1, input_tensor2 = input_tensor1 + noise, input_tensor2 + noise

            target_tensor = _target.to(device)

            output, loss = train(input_tensor1, input_tensor2, target_tensor, model,
             model_optimizer, criterion)
            
            accuracy = (output.argmax(-1) == target_tensor).float().mean()

            print_loss_total_train += loss
            plot_loss_total_train += loss
            print_acc_total_train += accuracy
            plot_acc_total_train += accuracy
            

            global_step = epoch * len(embedded) + local_step

            if global_step % print_every == 0:
                print_loss_avg_train = print_loss_total_train / print_every
                print_loss_total_train = 0
                
                print('%s (%d %d%%) train_loss = %.4f' % (timeSince(start, global_step / total_steps),
                                             global_step, global_step / total_steps * 100, print_loss_avg_train))


        plot_loss_avg_train = plot_loss_total_train / len(embedded)
        plot_losses_train.append(plot_loss_avg_train)
        
        plot_avg_acc_train = plot_acc_total_train / len(embedded)
        plot_acc_train.append(plot_avg_acc_train)
        
        plot_loss_total_train = 0
        plot_acc_total_train = 0
        
        train_epochs.append(epoch)


        if epoch % 5 == 0:


            for (_input1, _input2, _target) in val_embedded:

                input_tensor1 = _input1.to(device)
                input_tensor2 = _input2.to(device)
                target_tensor = _target.to(device)

                output, loss = _eval(input_tensor1, input_tensor2, target_tensor, model,
                             model_optimizer, criterion)

                accuracy = (output.argmax(-1) == target_tensor).float().mean()

                print_loss_total_val += loss
                plot_loss_total_val += loss
                print_acc_total_val += accuracy
                plot_acc_total_val += accuracy


            print_loss_avg_val = print_loss_total_val / len(val_embedded)
            print_loss_total_val = 0
            
            print_avg_acc = print_acc_total_val/ len(val_embedded)
            print_acc_total_val = 0
           
            print('val_loss = %.4f acc = %.4f' % (print_loss_avg_val, print_avg_acc))

            plot_loss_avg_val = plot_loss_total_val / len(val_embedded)
            plot_avg_acc_val = plot_acc_total_val / len(val_embedded)
            
            
            plot_losses_val.append(plot_loss_avg_val)
            plot_acc_val.append(plot_avg_acc_val)
            
            plot_loss_total_val = 0
            plot_acc_total_val = 0
            
            val_epochs.append(epoch)
       
        scheduler.step()

