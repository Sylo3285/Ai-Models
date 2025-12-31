import torch

in_channels = 1
epochs = 5
batch_size = 32
learning_rate = 0.001
device = "cuda"  if torch.cuda.is_available() else "cpu"
