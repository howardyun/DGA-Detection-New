import torch
import torch.nn as nn
import torch.nn.functional as F


class MITModel(nn.Module):
    def __init__(self, max_index, embedding_dimension):
        super(MITModel, self).__init__()

        # Embedding层
        self.embedding = nn.Embedding(num_embeddings=max_index,
                                      embedding_dim=embedding_dimension,
                                      padding_idx=0)

        # 卷积层
        # padding='same'在kersa表达的意思应该和pytorch的padding=1相同
        self.conv = nn.Conv1d(in_channels=embedding_dimension,
                              out_channels=128,
                              kernel_size=3,
                              padding=1)

        # 池化层
        self.max_pool = nn.MaxPool1d(kernel_size=2, padding=1)

        # LSTM层
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=64,
                            batch_first=True)

        # 全连接层
        self.fc = nn.Linear(64, 1)
        pass

    def forward(self, x):
        # 输入
        x = self.embedding(x)

        # 卷积层，这里可能要改改，pytorch希望channel在最后一个维度
        x = F.relu(self.conv(x.permute(0, 2, 1)))

        # 池化层
        x = self.max_pool(x)

        # LSTM层
        lstm_out, _ = self.lstm(x)

        # 选LSTM时序最后一次
        lstm_out = lstm_out[:, -1, :]

        # 全连接层
        x = torch.sigmoid(self.fc(lstm_out))

        return x
        pass

    pass


class MITMultiModel(nn.Module):
    def __init__(self, max_index, embedding_dimension, num_classes):
        super(MITMultiModel, self).__init__()

        # Embedding层
        self.embedding = nn.Embedding(num_embeddings=max_index,
                                      embedding_dim=embedding_dimension,
                                      padding_idx=0)

        # 卷积层
        # padding='same'在kersa表达的意思应该和pytorch的padding=1相同
        self.conv = nn.Conv1d(in_channels=embedding_dimension,
                              out_channels=128,
                              kernel_size=3,
                              padding=1)

        # 池化层
        self.max_pool = nn.MaxPool1d(kernel_size=2, padding=1)

        # LSTM层
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=64,
                            batch_first=True)

        # 全连接层
        self.fc = nn.Linear(64, num_classes)
        pass

    def forward(self, x):
        # 输入
        x = self.embedding(x)

        # 卷积层，这里可能要改改，pytorch希望channel在最后一个维度
        x = F.relu(self.conv(x.permute(0, 2, 1)))

        # 池化层
        x = self.max_pool(x)

        # LSTM层
        lstm_out, _ = self.lstm(x)

        # 选LSTM时序最后一次
        lstm_out = lstm_out[:, -1, :]

        # 全连接层
        return self.fc(lstm_out)
        # x = torch.softmax(self.fc(lstm_out), dim=1)
        # return x
        pass

    pass
