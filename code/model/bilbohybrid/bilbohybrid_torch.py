import torch
import torch.nn as nn
import torch.nn.functional as F


class BilBoHybridModel(nn.Module):
    """
    ann cnn lstm的组合模型
    """

    def __init__(self, max_index, embedding_dimension, num_conv_filters,
                 max_features=256):
        super(BilBoHybridModel, self).__init__()
        self.num_conv_filters = num_conv_filters

        # CNN
        self.embeddingCNN = nn.Embedding(num_embeddings=max_index,
                                         embedding_dim=embedding_dimension,
                                         padding_idx=0)

        self.conv2 = nn.Conv1d(in_channels=embedding_dimension, out_channels=num_conv_filters,
                               kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=embedding_dimension, out_channels=num_conv_filters,
                               kernel_size=3)
        self.conv4 = nn.Conv1d(in_channels=embedding_dimension, out_channels=num_conv_filters,
                               kernel_size=4)
        self.conv5 = nn.Conv1d(in_channels=embedding_dimension, out_channels=num_conv_filters,
                               kernel_size=5)
        self.conv6 = nn.Conv1d(in_channels=embedding_dimension, out_channels=num_conv_filters,
                               kernel_size=6)

        self.pool2 = nn.AdaptiveMaxPool1d(1)
        self.pool3 = nn.AdaptiveMaxPool1d(1)
        self.pool4 = nn.AdaptiveMaxPool1d(1)
        self.pool5 = nn.AdaptiveMaxPool1d(1)
        self.pool6 = nn.AdaptiveMaxPool1d(1)

        self.densecnn = nn.Linear(num_conv_filters * 5, num_conv_filters)
        self.dropoutcnnmid = nn.Dropout(0.5)
        self.dropoutcnn = nn.Dropout(0.5)

        # LSTM
        self.embeddingLSTM = nn.Embedding(num_embeddings=max_features,
                                          embedding_dim=256,
                                          padding_idx=0)

        self.lstm = nn.LSTM(input_size=256,
                            hidden_size=256,
                            batch_first=True)

        # 防止过拟合
        self.dropoutlstm = nn.Dropout(0.5)

        # ANN
        # self.extradense = nn.Linear(num_conv_filters * 5 + 256, 100)
        self.extradense = nn.Linear(num_conv_filters + 256, 100)
        self.output = nn.Linear(100, 1)

        self.dropoutsemifinal = nn.Dropout(0.5)
        self.dropoutfinal = nn.Dropout(0.5)
        pass

    def forward(self, x):
        x_cnn = self.embeddingCNN(x)
        x_lstm = self.embeddingLSTM(x)

        # CNN
        x2 = F.relu(self.conv2(x_cnn.permute(0, 2, 1)))
        x3 = F.relu(self.conv3(x_cnn.permute(0, 2, 1)))
        x4 = F.relu(self.conv4(x_cnn.permute(0, 2, 1)))
        x5 = F.relu(self.conv5(x_cnn.permute(0, 2, 1)))
        x6 = F.relu(self.conv6(x_cnn.permute(0, 2, 1)))

        x2 = self.pool2(x2).squeeze()
        x3 = self.pool3(x3).squeeze()
        x4 = self.pool4(x4).squeeze()
        x5 = self.pool5(x5).squeeze()
        x6 = self.pool6(x6).squeeze()

        # 处理张量问题
        if x2.size() == torch.Size([self.num_conv_filters]):
            # 模态融合,此时张量大小为[num_conv_filters],缺少了一个维度
            x2 = torch.unsqueeze(x2, dim=0)
            x3 = torch.unsqueeze(x3, dim=0)
            x4 = torch.unsqueeze(x4, dim=0)
            x5 = torch.unsqueeze(x5, dim=0)
            x6 = torch.unsqueeze(x6, dim=0)
            x_cnn = torch.cat([x2, x3, x4, x5, x6], dim=1)
            pass
        else:
            # 模态融合,此时张量大小为[batch_size, num_conv_filters]
            x_cnn = torch.cat([x2, x3, x4, x5, x6], dim=1)
            pass

        x_cnn = self.dropoutcnnmid(x_cnn)
        x_cnn = F.relu(self.densecnn(x_cnn))
        x_cnn = self.dropoutcnn(x_cnn)

        # LSTM
        # _, (x_lstm, _) = self.lstm(x_lstm)
        lstm_out, _ = self.lstm(x_lstm)

        lstm_out = lstm_out[:, -1, :]

        x_lstm = self.dropoutlstm(lstm_out)

        # ANN
        # x_combined = torch.cat([x_cnn, x_lstm[-1, :]], dim=1)
        x_combined = torch.cat([x_cnn, x_lstm], dim=1)

        x = self.dropoutsemifinal(x_combined)
        x = F.relu(self.extradense(x))
        x = self.dropoutfinal(x)
        x = F.relu(self.output(x))

        return torch.sigmoid(x)
        pass

    pass


class BBYBMultiModel(nn.Module):
    def __init__(self, max_index, embedding_dimension, num_conv_filters,
                 max_features=256, num_classes=2):
        super(BBYBMultiModel, self).__init__()
        self.num_conv_filters = num_conv_filters
        self.num_classes = num_classes

        # CNN
        self.embeddingCNN = nn.Embedding(num_embeddings=max_index,
                                         embedding_dim=embedding_dimension,
                                         padding_idx=0)

        self.conv2 = nn.Conv1d(in_channels=embedding_dimension, out_channels=num_conv_filters,
                               kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=embedding_dimension, out_channels=num_conv_filters,
                               kernel_size=3)
        self.conv4 = nn.Conv1d(in_channels=embedding_dimension, out_channels=num_conv_filters,
                               kernel_size=4)
        self.conv5 = nn.Conv1d(in_channels=embedding_dimension, out_channels=num_conv_filters,
                               kernel_size=5)
        self.conv6 = nn.Conv1d(in_channels=embedding_dimension, out_channels=num_conv_filters,
                               kernel_size=6)

        self.pool2 = nn.AdaptiveMaxPool1d(1)
        self.pool3 = nn.AdaptiveMaxPool1d(1)
        self.pool4 = nn.AdaptiveMaxPool1d(1)
        self.pool5 = nn.AdaptiveMaxPool1d(1)
        self.pool6 = nn.AdaptiveMaxPool1d(1)

        self.densecnn = nn.Linear(num_conv_filters * 5, num_conv_filters)
        self.dropoutcnnmid = nn.Dropout(0.5)
        self.dropoutcnn = nn.Dropout(0.5)

        # LSTM
        self.embeddingLSTM = nn.Embedding(num_embeddings=max_features,
                                          embedding_dim=256,
                                          padding_idx=0)

        self.lstm = nn.LSTM(input_size=256,
                            hidden_size=256,
                            batch_first=True)

        # 防止过拟合
        self.dropoutlstm = nn.Dropout(0.5)

        # ANN
        # self.extradense = nn.Linear(num_conv_filters * 5 + 256, 100)
        self.extradense = nn.Linear(num_conv_filters + 256, 100)
        self.output = nn.Linear(100, num_classes)

        self.dropoutsemifinal = nn.Dropout(0.5)
        self.dropoutfinal = nn.Dropout(0.5)
        pass

    def forward(self, x):
        x_cnn = self.embeddingCNN(x)
        x_lstm = self.embeddingLSTM(x)

        # CNN
        x2 = F.relu(self.conv2(x_cnn.permute(0, 2, 1)))
        x3 = F.relu(self.conv3(x_cnn.permute(0, 2, 1)))
        x4 = F.relu(self.conv4(x_cnn.permute(0, 2, 1)))
        x5 = F.relu(self.conv5(x_cnn.permute(0, 2, 1)))
        x6 = F.relu(self.conv6(x_cnn.permute(0, 2, 1)))

        x2 = self.pool2(x2).squeeze()
        x3 = self.pool3(x3).squeeze()
        x4 = self.pool4(x4).squeeze()
        x5 = self.pool5(x5).squeeze()
        x6 = self.pool6(x6).squeeze()

        # 处理张量问题
        if x2.size() == torch.Size([self.num_conv_filters]):
            # 模态融合,此时张量大小为[num_conv_filters],缺少了一个维度
            x2 = torch.unsqueeze(x2, dim=0)
            x3 = torch.unsqueeze(x3, dim=0)
            x4 = torch.unsqueeze(x4, dim=0)
            x5 = torch.unsqueeze(x5, dim=0)
            x6 = torch.unsqueeze(x6, dim=0)
            x_cnn = torch.cat([x2, x3, x4, x5, x6], dim=1)
            pass
        else:
            # 模态融合,此时张量大小为[batch_size, num_conv_filters]
            x_cnn = torch.cat([x2, x3, x4, x5, x6], dim=1)
            pass

        x_cnn = self.dropoutcnnmid(x_cnn)
        x_cnn = F.relu(self.densecnn(x_cnn))
        x_cnn = self.dropoutcnn(x_cnn)

        # LSTM
        # _, (x_lstm, _) = self.lstm(x_lstm)
        lstm_out, _ = self.lstm(x_lstm)

        lstm_out = lstm_out[:, -1, :]

        x_lstm = self.dropoutlstm(lstm_out)

        # ANN
        # x_combined = torch.cat([x_cnn, x_lstm[-1, :]], dim=1)
        x_combined = torch.cat([x_cnn, x_lstm], dim=1)

        x = self.dropoutsemifinal(x_combined)
        x = F.relu(self.extradense(x))
        x = self.dropoutfinal(x)
        x = F.relu(self.output(x))
        return x
        # return torch.softmax(x, dim=1)
        pass
    pass
