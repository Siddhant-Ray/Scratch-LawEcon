from __future__ import annotations

import math
import time

import torch


class DatasetManager(torch.utils.data.Dataset):
    def __init__(self, list_of_sent1, list_of_sent2, class_labels):
        self.list_of_sent1 = list_of_sent1
        self.list_of_sent2 = list_of_sent2
        self.class_labels = class_labels

    # get one sample
    def __getitem__(self, idx):

        input_tensor1 = torch.from_numpy(self.list_of_sent1[idx]).float()
        input_tensor2 = torch.from_numpy(self.list_of_sent2[idx]).float()
        target_tensor = torch.tensor(self.class_labels[idx]).unsqueeze(0).float()
        # target_tensor = torch.tensor(self.class_labels[idx]).float()

        return input_tensor1, input_tensor2, target_tensor

    def __len__(self):
        return len(self.list_of_sent1)


def train(
    input_tensor1, input_tensor2, target_tensor, model, model_optimizer, criterion
):
    model.train()

    model_optimizer.zero_grad()

    input_length = input_tensor1.shape
    target_length = target_tensor.shape
    output = model(input_tensor1, input_tensor2)

    loss = criterion(output, target_tensor)
    loss.backward()

    model_optimizer.step()

    return output, loss.item()


def evaluate(
    input_tensor1, input_tensor2, target_tensor, model, model_optimizer, criterion
):
    model.eval()

    with torch.no_grad():

        input_length = input_tensor1.shape
        target_length = target_tensor.shape
        output = model(input_tensor1, input_tensor2)

        loss = criterion(output, target_tensor)

    return output, loss.item()


def asMinutes(secs):
    mins = math.floor(secs / 60)
    secs -= mins * 60
    return "%dm %ds" % (mins, secs)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (- %s)" % (asMinutes(s), asMinutes(rs))
