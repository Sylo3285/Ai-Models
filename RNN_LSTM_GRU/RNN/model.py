import torch.nn as nn
import torch
from config import *


class SimpleRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            batch_first=True,
            nonlinearity="tanh"
        )
        self.fc = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x, hidden=None):
        _, hidden = self.rnn(x, hidden)
        logits = self.fc(hidden[-1])
        return logits, hidden
    
def load_model(model_path):
    model = SimpleRNN().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Model saved files not found")
    model.eval()
    return model