import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_SIZE = 1
HIDDEN_SIZE = 128
OUTPUT_SIZE = 3
SEQ_LEN = 3          # ← previous N days
NUM_EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3

movement_to_label = {
    "down": 0,
    "same": 1,
    "up": 2
}

label_to_movement = {v: k for k, v in movement_to_label.items()}