import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# class TempCNN(nn.Module):

class MLP(nn.Module):
    def __init__(self, input_size, n_class, dropout_rate=0.5, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.flatten = nn.Flatten()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.cl = nn.Linear(256, n_class)
        # self.cl = nn.Linear(256, 1)

    def forward(self, inputs):
        output = self.flatten(inputs)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.cl(output)
        return output
        # return nn.Sigmoid()(output)
