import torch.nn as nn
from configs.config import *

class ANN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.GELU(),
            nn.LayerNorm(HIDDEN_DIM),
            nn.Dropout(DROPOUT),

            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.LayerNorm(HIDDEN_DIM),
            nn.Dropout(DROPOUT),

            nn.Linear(HIDDEN_DIM, num_classes),
        )

    def forward(self, x):
        return self.net(x)