import torch.nn as nn
from config import *

class ANN(nn.Module):

    def __init__(self,input_dim,num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM,HIDDEN_DIM//2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM//2,HIDDEN_DIM//4), 
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM//4,num_classes)
        )

    def forward(self, x):
        return self.net(x)
