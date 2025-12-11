import torch

#   model configs
inp_size = (128, 128)  # input image size

#   training configs
learning_rate = 0.01
num_epochs = 200
batch_size = 256
momentum = 0.9
weight_decay = 1e-4  # L2 regularization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

