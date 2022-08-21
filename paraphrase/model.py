from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

"""class SNNLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.normal_(self.fc.weight, std=math.sqrt(1 / self.fc.weight.shape[1]))
        nn.init.zeros_(self.fc.bias)"""


class SNNLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        nn.init.normal_(self.fc.weight, std=math.sqrt(
            1 / self.fc.weight.shape[1]))
        nn.init.zeros_(self.fc.bias)

    def forward(self, inputs):
        return self.fc(inputs)


class SimilarityNN(nn.Module):
    """Simple NN architecture"""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.transform = nn.Sequential(
            nn.AlphaDropout(0.2),
            SNNLinear(input_size, hidden_size),
            nn.SELU(),
            nn.AlphaDropout(0.2),
            SNNLinear(hidden_size, hidden_size // 2),
        )

        self.cosine_sim = nn.CosineSimilarity(dim=1)

        self.combination = nn.Sequential(
            nn.SELU(),
            nn.AlphaDropout(0.2),
            SNNLinear(hidden_size // 2, output_size),
        )

    def forward(self, input1, input2):
        c1 = self.transform(input1)
        c2 = self.transform(input2)
        # return self.cosine_sim(c1, c2)/ 2 + 0.5
        return self.combination(c1 + c2)


class LinearSimilarityNN(nn.Module):
    """Simple NN architecture"""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(input_size, hidden_size, bias=True),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2, bias=True),
        )
        self.cosine_sim = nn.CosineSimilarity(dim=1)

        self.combination = nn.Sequential(
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, input1, input2):
        c1 = self.transform(input1)
        c2 = self.transform(input2)
        # return self.cosine_sim(c1, c2)/ 2 + 0.5
        return self.combination(c1 + c2)
