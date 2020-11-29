import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self, vocab_size=10002, embedding_dim=100, vector_len=80, hidden_size=64):
        super(DNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # output shape: [batch size, vector len, embedding_dim]

        self.flatten = nn.Flatten()
        self.linear_01 = nn.Linear(vector_len * embedding_dim, hidden_size)
        self.norm_01 = nn.BatchNorm1d(hidden_size)

        self.linear_02 = nn.Linear(hidden_size, hidden_size)
        self.norm_02 = nn.BatchNorm1d(hidden_size)

        self.linear_03 = nn.Linear(hidden_size, 1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        embedding_vec = self.embedding(x)

        linear_vec_1 = self.relu(self.norm_01(self.linear_01(self.flatten(embedding_vec))))
        linear_vec_2 = self.relu(self.norm_02(self.linear_02(linear_vec_1)))
        out = self.sigmoid((self.linear_03(linear_vec_2)))

        return out


class LSTM(nn.Module):
    def __init__(self, vocab_size=10002, embedding_dim=100, vector_len=80, unit_num=128):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # output shape: [batch size, vector len, embedding_dim]

        self.lstm = nn.LSTM(embedding_dim, unit_num, num_layers=2, batch_first=True)  # output shape: [batch size, vector len, unit num]

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(unit_num, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedding_vec = self.embedding(x)
        lstm_vec = self.lstm(embedding_vec)[0]
        out = self.sigmoid(self.linear(self.flatten(lstm_vec[:, -1, :])))  # use last cell output of lstm layer

        return out


class Res_CNN(nn.Module):
    def __init__(self, vocab_size=10002, embedding_dim=100, vector_len=80, filter_size=6):
        super(Res_CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # output shape: [batch size, vector len, embedding_dim]

        self.pad_01_1 = nn.ZeroPad2d(3)
        self.conv_01_1 = nn.Conv2d(1, filter_size, 7, 2)
        self.norm_01_1 = nn.BatchNorm2d(filter_size)

        self.pad_02_1 = nn.ZeroPad2d(1)
        self.conv_02_1 = nn.Conv2d(filter_size, filter_size, 3)
        self.norm_02_1 = nn.BatchNorm2d(filter_size)

        self.pad_02_2 = nn.ZeroPad2d(1)
        self.conv_02_2 = nn.Conv2d(filter_size, filter_size, 3)
        self.norm_02_2 = nn.BatchNorm2d(filter_size)

        self.pad_03_1 = nn.ZeroPad2d((0, 0, 1, 1))
        self.conv_03_1 = nn.Conv2d(filter_size, 1, (3, int(embedding_dim / 2)))
        self.norm_03_1 = nn.BatchNorm2d(1)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(int(vector_len / 2), 1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        embedding_vec = self.embedding(x)  # output shape: [batch size, vector len, embedding_dim]

        conv_vec_0 = torch.unsqueeze(embedding_vec, 1)  # convert shape from [batch size, vector len, embedding dim] to [batch size, 1, vector len, embedding dim]

        # down convolutional layer.
        conv_vec_1 = self.relu(self.norm_01_1(self.conv_01_1(self.pad_01_1(conv_vec_0))))

        # one residual block(skip layer)
        conv_vec_2 = self.relu(self.norm_02_1(self.conv_02_1(self.pad_02_1(conv_vec_1))))
        conv_vec_3 = self.relu(self.norm_02_2(self.conv_02_2(self.pad_02_2(conv_vec_2)))) + conv_vec_1

        conv_vec_4 = self.relu(self.norm_03_1(self.conv_03_1(self.pad_03_1(conv_vec_3))))

        out = self.sigmoid(self.linear(self.flatten(conv_vec_4)))

        return out


if __name__ == "__main__":
    rnn = nn.LSTM(10, 20, 2)
    input = torch.randn(5, 3, 10)
    h0 = torch.randn(2, 3, 20)
    c0 = torch.randn(2, 3, 20)
    output, (hn, cn) = rnn(input, (h0, c0))
    print(output.shape)  # [5, 3, 20]
