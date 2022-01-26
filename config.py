import torch

# gpu setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# model parameter setting
BATCH_SIZE = 512
n_layers = 6
d_model = 512
ffn_hidden = 2024
n_heads = 8
dropout = 0.1

# optimizer parameter setting
warmup = 2000
factor = 1
epoch = 1000
inf = float('inf')
