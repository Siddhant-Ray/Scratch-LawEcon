import json, pickle
import os, sys, time, math

import torch
import torch.nn as nn
import numpy as np
from torch import optim

from utils import DatasetManager
from utils import train, evaluate
from utils import asMinutes, timeSince

from model import SimilarityNN, LinearSimilarityNN

torch.manual_seed(0)
np.random.seed(0)

import random
random.seed(0)

from torch.optim.lr_scheduler import StepLR
from tqdm.notebook import tqdm

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score

# Load full dataset with combined NLI pairs
with open('paraphrase/data/embeddings_1.pkl', "rb") as em1:
    stored_data_1 = pickle.load(em1)
with open('paraphrase/data/embeddings_2.pkl', "rb") as em2:
    stored_data_2 = pickle.load(em2)
with open('paraphrase/data/labels.pkl', "rb") as lbl:
    stored_labels = pickle.load(lbl)

# Load only mprc dataset 
with open('paraphrase/data/mprc_embeddings_1.pkl', "rb") as _em1:
    stored_data_mprc_1 = pickle.load(_em1)
with open('paraphrase/data/mprc_embeddings_2.pkl', "rb") as _em2:
    stored_data_mprc_2 = pickle.load(_em2)
with open('paraphrase/data/mprc_labels.pkl', "rb") as _lbl:
    stored_labels_mprc = pickle.load(_lbl)

#print(len(stored_data_1['embeddings']), type(stored_data_1['embeddings']))
#print(len(stored_data_2['embeddings']), type(stored_data_2['embeddings']))
#print(len(stored_labels['labels']), type(stored_labels['labels']))

with open('paraphrase/configs/config.json', 'r') as f:
    config = json.load(f)

## Hyperparameters
learning_rate = config['learning_rate']
hidden_size = config['hidden_neurons']
input_size1 = stored_data_mprc_1['embeddings'].shape[1]
input_size2 = stored_data_mprc_2['embeddings'].shape[1]
output_size = config['output_neurons']
epochs = config['num_epochs']
batch_size = config['batch_size']
weight_decay = config['weight_decay']

dataset = DatasetManager(stored_data_mprc_1['embeddings'], stored_data_mprc_2['embeddings'], stored_labels_mprc['labels'])

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
            batch_size=batch_size,
            shuffle=True,
)
val_dataloader = torch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def trainIters(model, n_iters, embedded, val_embedded,learning_rate, writer,
                print_every = 1):
    start = time.time()
    print_loss_total_train = 0  # Reset every print_every
    print_loss_total_val = 0  # Reset every print_every
    
    print_acc_total_train = 0
    print_acc_total_val = 0
    
    train_epochs = []
    val_epochs = []

    #TODO: Try ADAM
    model_optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-5)
    
    #TODO: Learning rate scheduler
    scheduler = StepLR(model_optimizer, step_size = n_iters//2, gamma=0.1)

    #criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCELoss()

    
    total_steps = n_iters*len(embedded)
    
    for epoch in range(1, n_iters + 1):
        
        pred_temp = 0
        true_temp = 0
        y_true = []
        y_pred = []

        print("Training phase", "=" * 25, ">")

        for local_step, (_input1, _input2, _target) in enumerate(embedded, 1):

            input_tensor1 = _input1.to(device)
            input_tensor2 = _input2.to(device)

            noise = torch.randn_like(input_tensor1) * 1e-3 
            # input_tensor1, input_tensor2 = input_tensor1, input_tensor2 + noise

            target_tensor = _target.to(device)

            output, loss = train(input_tensor1, input_tensor2, target_tensor, model,
             model_optimizer, criterion)
            
            accuracy = ((output >  0.5) == target_tensor).float().mean()

            print_loss_total_train += loss
            print_acc_total_train += accuracy

            #predictions for f1 score
            pred_temp = (output.clone() > 0.5).float().cpu().detach().numpy()
            true_temp = target_tensor.cpu().detach().numpy()
            # print("target batch",true_temp)
            # print("predicted batch",pred_temp)
            # breakpoint()

            for item in pred_temp:
                y_pred.append(item)

            for item in true_temp:
                y_true.append(item)
            
            global_step = epoch * len(embedded) + local_step

            '''if global_step % print_every == 0:
                print_loss_avg_train = print_loss_total_train / print_every
                print_loss_total_train = 0
                
                print('%s (%d %d%%) train_loss = %.4f' % (timeSince(start, global_step / total_steps),
                                             global_step, global_step / total_steps * 100, print_loss_avg_train))
        '''
        print_loss_avg_train = print_loss_total_train/ len(embedded)
        print_loss_total_train = 0 
        print_acc_avg_train = print_acc_total_train/ len(embedded)
        print_acc_total_train = 0

        print('epoch = %d train_loss = %.4f train_acc = %.4f' %(epoch, print_loss_avg_train, print_acc_avg_train))

        writer.add_scalar("Loss/train", print_loss_avg_train, epoch)
        writer.add_scalar("Acc/train", print_acc_avg_train, epoch)

        f1score = f1_score(y_true, y_pred)
        print("train_F1 score for epoch = {epoch}".format(epoch = epoch), "is", f1score)
        # breakpoint()
        writer.add_scalar("F1/train", f1score, epoch)
        train_epochs.append(epoch)

        pred_temp = 0
        true_temp = 0
        y_true = []
        y_pred = []

        if epoch % 5 == 0:
            print()
            print("Validation phase", "=" * 25, ">")

            for (_input1, _input2, _target) in val_embedded:

                input_tensor1 = _input1.to(device)
                input_tensor2 = _input2.to(device)
                target_tensor = _target.to(device)

                output, loss = evaluate(input_tensor1, input_tensor2, target_tensor, model,
                             model_optimizer, criterion)

                accuracy = ((output >  0.5) == target_tensor).float().mean()

                print_loss_total_val += loss
                print_acc_total_val += accuracy 

                pred_temp = (output.clone() > 0.5).float().cpu().detach().numpy()
                true_temp = target_tensor.cpu().detach().numpy()

                for item in pred_temp:
                    y_pred.append(item)

                for item in true_temp:
                    y_true.append(item)

            print_loss_avg_val = print_loss_total_val / len(val_embedded)
            print_loss_total_val = 0
            
            print_avg_acc_val = print_acc_total_val/ len(val_embedded)
            print_acc_total_val = 0
           
            print('epoch = %d val_loss = %.4f val_acc = %.4f' %(epoch, print_loss_avg_val, print_avg_acc_val))

            writer.add_scalar("Loss/val", print_loss_avg_val, epoch)
            writer.add_scalar("Acc/val", print_avg_acc_val, epoch)

            f1score = f1_score(y_true, y_pred)
            print("val_f1 score for epoch = {epoch}".format(epoch = epoch), "is", f1score)

            writer.add_scalar("F1/val", f1score, epoch)

            val_epochs.append(epoch)
       
        scheduler.step()
        print()
    writer.close()

def run_model():

    hidden_size_model = hidden_size
    input_size_model = input_size1
    output_size_model = output_size 
    epochs_model = epochs
    learning_rate_model = learning_rate 

    choose_model = sys.argv

    if choose_model[1] == "selu":
        print("This is the SimilarityNN WITH self normalization")
        writer = SummaryWriter("snnclassifiermetrics")
        model = SimilarityNN(input_size_model, hidden_size_model, output_size_model).to(device)
        trainIters(model, epochs_model, train_dataloader, val_dataloader, learning_rate_model,
                    writer,print_every= 1)
    elif choose_model[1] == "lin":
        print("This is the SimilarityNN WITHOUT self normalization")
        writer = SummaryWriter("linearclassifiermetrics")
        model = LinearSimilarityNN(input_size_model, hidden_size_model, output_size_model).to(device)
        trainIters(model, epochs_model, train_dataloader, val_dataloader, learning_rate_model,
                    writer,print_every= 1)

if __name__ == '__main__':
    run_model()
