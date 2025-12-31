import torch

# Configuration settings for the ANN model
device = "cuda" if torch.cuda.is_available() else "cpu"
input_size = 2
hidden_size = 256
output_size = 1

#training parameters
num_epochs = 50
lr = 0.001
batch_size = 100000
