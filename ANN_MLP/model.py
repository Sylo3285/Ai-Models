import torch.nn as nn
from config import *

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)

        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

        self.layer2 = nn.Linear(hidden_size, output_size)

        self.act = nn.GELU()

        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        out = self.act(self.bn1(self.layer1(x)))
        out = self.act(self.bn2(self.hidden1(out)))
        out = self.layer2(out)
        return out
