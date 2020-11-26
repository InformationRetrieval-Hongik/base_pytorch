import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.module):
    def __init__(self, vocab_size, embedding_dim):
        super(DNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
