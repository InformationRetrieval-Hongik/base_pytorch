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
    """
    Resnet의 Skip-Layer 컨셉을 가져와 적용해봤습니다.
    간단한 DNN 모델도 nsmc 데이터셋에서 Over-fitting 나는 문제가 있어서, Residual Block을 하나만 사용하여 구현했습니다.
    """

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


class Bi_GRU_layer_2(nn.Module):
    """
    def _init_state 함수부분은  https://wikidocs.net/60691  사이트에 존재하는 기존의 GRU model 에서 참조하였습니다
    이외의 부분은  Pytorch의 Documentation을 참고하여 input,output을 염두에 두고 구현했습니다.
    """

    def __init__(self, vocab_size):
        super(Gru, self).__init__()
        self.n_layers = 2
        self.vocab_size = vocab_size
        self.embedding_layer = nn.Embedding(self.vocab_size, embedding_dim=100)
        self.hidden_size = 80
        self.gru = nn.GRU(100, self.hidden_size, num_layers=self.n_layers, batch_first=True, bidirectional=True, dropout=0.3)
        self.linear = nn.Linear(self.hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedding = self.embedding_layer(x)
        h_0 = self._init_state(batch_size=x.size(0))
        gru_out, _ = self.gru(embedding, h_0)

        out = self.linear(gru_out[:, -1, :])
        return self.sigmoid(out)

    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers * 2, batch_size, self.hidden_size).zero_()


class Attention(nn.Module):
    """
    Attention 코드는 아래 주소의 수식과 설명을 보고 구현하였으며
    decoder 용 끝단을 classification에 맞추기 위해 concat이 아닌 attention만 반환합니다.
    Attention 메커니즘 출처 https://wikidocs.net/22893
    """

    def __init__(self, hidden_size, bidirectional=False, score_function="dot"):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.softmax = nn.Softmax()
        self.score_function = score_function

        # if score_function == 'general':

        if bidirectional:  # Is lstm bidirectional or not
            self.direction = 2
        else:
            self.direction = 1

    def forward(self, lstm_out, hidden):
        # lstm output shape [batch, seq_length, num_direction * hidden_size]
        # lstm hidden shape [num_direction, batch, hidden_size]
        seq_length = lstm_out.shape[1]
        hidden = hidden.view(-1, self.direction * self.hidden_size)  # make hidden shape : [batch, num_direction * hidden_size]

        # calculate Attention score
        # score(st,hi) = st.T * hi

        if self.score_function == "dot":
            score = torch.bmm(lstm_out, hidden.unsqueeze(2))
        elif self.score_function == "scaled_dot":
            score = torch.bmm(lstm_out, hidden.unsqueeze(2)) / seq_length

        at = self.softmax(score)  # et : [st*h1,....,st*hN], at : softmax(et) [batch, seq_length,1]

        attention = torch.bmm(at.transpose(1, 2), lstm_out)  # attention : [batch,1,num_direction * hidden_size]
        return attention.squeeze(1)  # return : [batch,num_direction * hidden_size]


class BILSTM_withAttention2layer(nn.Module):
    def __init__(self, vocab_size=1002, embedding_dim=10, vector_len=80, unit_num=128):
        super(BILSTM_withAttention2layer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # output shape: [batch size, vector len, embedding_dim]
        self.lstm1 = nn.LSTM(embedding_dim, unit_num, bidirectional=True, batch_first=True, dropout=0.35)  # output shape: [batch size, vector len, unit num]

        self.lstm2 = nn.LSTM(256, unit_num, bidirectional=True, batch_first=True, dropout=0.35)
        self.atten1 = Attention(unit_num, bidirectional=True, score_function="scaled_dot")
        self.atten2 = Attention(unit_num, bidirectional=True, score_function="scaled_dot")

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(unit_num * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedding_vec = self.embedding(x)
        out1, (h_n, c_n) = self.lstm1(embedding_vec)

        att = self.atten1(out1, h_n).unsqueeze(1)

        for i in range(out1.shape[1] - 2, -1, -1):
            att = torch.cat((self.atten1(out1[:, :i, :], out1[:, i, :]).unsqueeze(1), att), 1)

        out2, (h_n, c_n) = self.lstm2(att)

        att = self.atten2(out2, h_n)
        out = self.sigmoid(self.linear(att))

        return out


class Multi_Channel_CNN(nn.Module):
    def __init__(self, vocab_size=10002, embedding_dim=100, vector_len=80, n_filters_a=64, n_filters_b=128, filter_sizes_a=[3, 4, 5], filter_sizes_b=[3, 4, 5], dropout=0.5):
        super(Double_CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv_list_a = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=n_filters_a, kernel_size=(fs_a, embedding_dim)) for fs_a in filter_sizes_a])

        self.conv_list_b = nn.ModuleList([nn.Conv2d(in_channels=n_filters_a, out_channels=n_filters_b, kernel_size=(fs_b, 1)) for fs_b in filter_sizes_b])

        self.fully_Connected = nn.Linear(len(filter_sizes_b) * n_filters_b, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch_size, maxLen]

        embedding_vec = self.embedding(x)
        # embedding_vec = [batch_size, maxLen, emb_dim]
        embedding_vec = embedding_vec.unsqueeze(1)
        # embedding_vec = embedding_vec.permute(0, 2, 1)
        # # embedding_vec = [batch_size, 1, maxLen, emb_dim]

        conv_vecs = [self.relu(conv(embedding_vec)) for conv in self.conv_list_a]
        # conv_vecs = [batch_size, n_filters, maxlen - filter_sizes[n] + 1, 1]
        # for conv in self.conv_vecs:
        #     print(conv.shape + "\n")

        conv_vecs = [self.relu(self.conv_list_b[idx](conv_vecs[idx])).squeeze(3) for idx in range(len(self.conv_list_b))]
        # conv_vecs = [batch_size, n_filters, maxlen - filter_sizes[n] + 1]

        pooled_vecs = [F.max_pool1d(conv_vec, conv_vec.shape[2]).squeeze(2) for conv_vec in conv_vecs]
        # pooled_vec = [batch_size, n_filters]

        concat = self.dropout(torch.cat(pooled_vecs, dim=1))
        # concat = [batch_size, n_filters * len(filter_sizes)]

        out = self.sigmoid(self.fully_Connected(concat))
        # out = [batch_size, 1]

        return out


if __name__ == "__main__":
    rnn = nn.LSTM(10, 20, 2)
    input = torch.randn(5, 3, 10)
    h0 = torch.randn(2, 3, 20)
    c0 = torch.randn(2, 3, 20)
    output, (hn, cn) = rnn(input, (h0, c0))
    print(output.shape)  # [5, 3, 20]
