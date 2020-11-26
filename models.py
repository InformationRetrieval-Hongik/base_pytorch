import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self, vocab_size=1002, embedding_dim=10, vector_len=80):
        super(DNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.flatten = nn.Flatten()
        self.linear_a = nn.Linear(vector_len * embedding_dim, vector_len * embedding_dim)
        self.linear_b = nn.Linear(vector_len * embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedding_vec = self.embedding(x)
        flat_vec = self.flatten(embedding_vec)
        linear_vec = self.linear_a(flat_vec)
        linear_vec = self.linear_b(linear_vec)
        out = self.sigmoid(linear_vec)

        return out
