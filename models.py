import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self, vocab_size=1002, embedding_dim=10, vector_len=80):
        super(DNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.flatten = nn.Flatten()
        self.linear_a = nn.Linear(vector_len * embedding_dim, 64)
        self.linear_b = nn.Linear(64, 64)
        self.linear_c = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedding_vec = self.embedding(x)
        linear_vec_a = self.relu(self.linear_a(self.flatten(embedding_vec)))
        linear_vec_b = self.relu(self.linear_b(linear_vec_a))
        out = self.sigmoid(self.linear_c(linear_vec_b))

        return out


class LSTM(nn.Module):
    def __init__(self, vocab_size=1002, embedding_dim=10, vector_len=80, unit_num=128):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # output shape: [batch size, vector len, embedding_dim]
        self.lstm = nn.LSTM(embedding_dim, unit_num, num_layers=2, batch_first=True)  # output shape: [batch size, vector len, unit num]
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(unit_num, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.dtype != torch.LongTensor:
            x = x.type(torch.LongTensor)
        embedding_vec = self.embedding(x)
        lstm_vec = self.lstm(embedding_vec)[0]
        out = self.sigmoid(self.linear(self.flatten(lstm_vec[:, -1, :])))

        return out


if __name__ == "__main__":
    rnn = nn.LSTM(10, 20, 2)
    input = torch.randn(5, 3, 10)
    h0 = torch.randn(2, 3, 20)
    c0 = torch.randn(2, 3, 20)
    output, (hn, cn) = rnn(input, (h0, c0))
    print(output.shape)  # [5, 3, 20]
