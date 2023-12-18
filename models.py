# -*- coding = utf-8 -*-
# @File : models.py
# @Software : PyCharm
import torch
from torch import nn

'''
nn.Embedding 是 PyTorch 中的一个类，它用于将一个大整数序列（比如文本中的单词序列）编码为一个低维稠密向量序列，这些向量可以被神经网络用于后续的处理任务。
在自然语言处理中，这个过程叫做“词嵌入”（word embedding）。
具体来说，nn.Embedding 可以将一个大整数序列（如单词 ID 序列）映射到一个低维稠密向量序列。
这个映射过程是通过将每个整数映射到一个对应的向量来实现的。这些向量可以在训练过程中学习得到，也可以使用预训练的词嵌入（如 GloVe、Word2Vec 等）。

在创建 nn.Embedding 实例时，需要指定两个参数 vocab_size 和 embedding_dim，它们分别代表词汇表大小和嵌入维度。
vocab_size 是词汇表中不同单词的数量，它通常是一个整数。embedding_dim 是指嵌入向量的维度，它是一个正整数。
通常情况下，embedding_dim 的值越大，嵌入向量的表示能力越强，但同时需要更多的计算资源和更多的训练数据来进行训练。
具体来说，nn.Embedding 会将一个大小为 (batch_size, sequence_length) 的输入张量（即整数序列）中的每个整数都映射为一个大小为 (embedding_dim,) 的嵌入向量。
因此，self.embedding_table 的大小为 (vocab_size, embedding_dim)，其中每行代表词汇表中的一个单词，每列代表嵌入向量的一个维度。
'''

VOCAB_SIZE = 15000  # 单词表大小
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GCNN(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=64, num_class=2):
        # embedding_dim: 单词表中每一个token的向量大小
        super(GCNN, self).__init__()

        self.embedding_table = nn.Embedding(vocab_size, embedding_dim)  # 实例化embedding_table 行vocab_size列embedding_dim
        nn.init.xavier_uniform_(self.embedding_table.weight)

        self.conv_A_1 = nn.Conv1d(embedding_dim, out_channels=32, kernel_size=15, stride=7)
        self.conv_B_1 = nn.Conv1d(embedding_dim, out_channels=32, kernel_size=15, stride=7)

        self.conv_A_2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=15, stride=7)
        self.conv_B_2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=15, stride=7)

        self.output_linear1 = nn.Linear(64, 128)
        self.output_linear2 = nn.Linear(128, num_class)

    def forward(self, word_index):
        #  基于句子单词ID输入得到分类logits输出

        # 1. 通过word_index得到word_embedding
        # word_index shape: [bs, max_seq_len]
        word_embedding = self.embedding_table(word_index)  # [bs, max_seq_len, embedding_dim]

        # 2. 两层门卷积模块
        word_embedding = word_embedding.transpose(1, 2)  # [bs, embedding_dim, max_seq_len]
        A = self.conv_A_1(word_embedding)
        B = self.conv_B_1(word_embedding)
        H = A * torch.sigmoid(B)  # [bs, 128, max_seq_len]

        A = self.conv_A_2(H)
        B = self.conv_B_2(H)
        H = A * torch.sigmoid(B)  # [bs, 64, max_seq_len]

        # 3. 池化并通过全连接层
        pool_output = torch.mean(H, dim=-1)  # 最后一维平均池化 [bs, 64]
        linear1_output = self.output_linear1(pool_output)
        logits = self.output_linear2(linear1_output)

        return logits


'''
nn.EmbeddingBag 是 PyTorch 中用于将文本序列转换为向量表示的一种层级。
它与 nn.Embedding 类似，都是将输入的离散化数据（如单词 ID）映射到一个连续的向量空间中，以便神经网络可以学习到单词之间的语义关系。
不同之处在于，nn.Embedding 只是将每个单词映射到一个固定的向量，而 nn.EmbeddingBag 则可以将一个句子中的所有单词向量进行平均或加权平均，生成一个代表整个句子的向量。
'''


class DNN(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=64, num_class=2):
        super(DNN, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, sparse=False)
        self.fc = nn.Linear(embedding_dim, num_class)

    def forward(self, token_index):
        embedding = self.embedding(token_index)  # [bs, embedding_dim]
        x = self.fc(embedding)
        return x


class GCNN_LSTM(nn.Module):
    def __init__(self, hidden_size, num_layers, vocab_size=VOCAB_SIZE, embedding_dim=256, num_class=2):
        # embedding_dim: 单词表中每一个token的向量大小
        super(GCNN_LSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding_table = nn.Embedding(vocab_size, embedding_dim)  # 实例化embedding_table 行vocab_size列embedding_dim
        nn.init.kaiming_normal_(self.embedding_table.weight)

        self.conv_A_1 = nn.Conv1d(embedding_dim, out_channels=128, kernel_size=15, stride=1)
        self.conv_B_1 = nn.Conv1d(embedding_dim, out_channels=128, kernel_size=15, stride=1)

        self.relu = nn.ReLU(inplace=True)

        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # Fully connected layer
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, num_class)

    def forward(self, word_index):
        word_embedding = self.embedding_table(word_index)  # [bs, max_seq_len, embedding_dim]

        # GCNN layers
        word_embedding = word_embedding.transpose(1, 2)  # [bs, embedding_dim, max_seq_len]
        A = self.conv_A_1(word_embedding)
        A = self.relu(A) + A

        B = self.conv_B_1(word_embedding)
        B = self.relu(B) + B
        out = A * torch.sigmoid(B)  # [bs, 128, max_seq_len]

        out = torch.mean(out, dim=-1)  # 最后一维平均池化 [bs, 128]

        # LSTM layers
        h0 = torch.zeros(self.num_layers, out.size(0), self.hidden_size).to(out.device)
        c0 = torch.zeros(self.num_layers, out.size(0), self.hidden_size).to(out.device)

        out, _ = self.lstm(out.unsqueeze(1), (h0, c0))

        # Fully connected layer
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out[:, -1, :])

        return out


if __name__ == '__main__':
    model = GCNN_LSTM(hidden_size=64, num_layers=3)
    input = torch.randint(0, 10, size=(32, 690))
    print(input.shape)
    output = model(input)
    # print(output)
    print(output.shape)
