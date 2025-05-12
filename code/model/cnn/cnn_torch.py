import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self, max_index, max_string_length, embedding_dimension,
                 num_conv_filters):
        """
        :param num_conv_filters: 卷积神经网络输出空间维度
        """
        super(CNNModel, self).__init__()
        self.num_conv_filters = num_conv_filters

        # Embedding层
        self.embeddingCNN = nn.Embedding(num_embeddings=max_index,
                                         embedding_dim=embedding_dimension,
                                         padding_idx=0)

        # 五层平行的卷积神经网络，卷积核不同
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

        # 全局池化层，kersa中使用的GlobalMaxPool1D求取最大值，但前面输入的卷积神经网络是一维的，最终得到一维最大数，
        # 因此可用torch的AdaptiveMaxPool1d(1)达到相同目的
        self.pool2 = nn.AdaptiveMaxPool1d(1)
        self.pool3 = nn.AdaptiveMaxPool1d(1)
        self.pool4 = nn.AdaptiveMaxPool1d(1)
        self.pool5 = nn.AdaptiveMaxPool1d(1)
        self.pool6 = nn.AdaptiveMaxPool1d(1)

        # 全连接层
        self.densecnn = nn.Linear(num_conv_filters * 5, num_conv_filters)
        self.dropoutcnnmid = nn.Dropout(0.5)
        self.dropoutcnn = nn.Dropout(0.5)

        # 输出层
        self.output = nn.Linear(num_conv_filters, 1)
        pass

    def forward(self, x):
        # 输入
        x = self.embeddingCNN(x)

        # 五层卷积层，这里可能要改改，pytorch希望channel在最后一个维度
        # x2 = F.relu(self.conv2(x))
        # x3 = F.relu(self.conv3(x))
        # x4 = F.relu(self.conv4(x))
        # x5 = F.relu(self.conv5(x))
        # x6 = F.relu(self.conv6(x))
        x2 = F.relu(self.conv2(x.permute(0, 2, 1)))
        x3 = F.relu(self.conv3(x.permute(0, 2, 1)))
        x4 = F.relu(self.conv4(x.permute(0, 2, 1)))
        x5 = F.relu(self.conv5(x.permute(0, 2, 1)))
        x6 = F.relu(self.conv6(x.permute(0, 2, 1)))

        # 池化层
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
            x = torch.cat([x2, x3, x4, x5, x6], dim=1)
            pass
        else:
            # 模态融合,此时张量大小为[batch_size, num_conv_filters]
            x = torch.cat([x2, x3, x4, x5, x6], dim=1)
            pass

        # 全连接层
        x = self.dropoutcnnmid(x)
        x = F.relu(self.densecnn(x))
        x = self.dropoutcnn(x)

        # 激活，输出
        x = torch.sigmoid(self.output(x))

        return x
        pass

    pass


class CNNMultiModel(nn.Module):
    def __init__(self, max_index, max_string_length, embedding_dimension,
                 num_conv_filters, num_classes):
        """
        :param num_conv_filters: 卷积神经网络输出空间维度
        """
        super(CNNMultiModel, self).__init__()
        self.num_conv_filters = num_conv_filters
        # 多分类标签
        self.num_classes = num_classes

        # Embedding层
        self.embeddingCNN = nn.Embedding(num_embeddings=max_index,
                                         embedding_dim=embedding_dimension,
                                         padding_idx=0)

        # 五层平行的卷积神经网络，卷积核不同
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

        # 全局池化层，kersa中使用的GlobalMaxPool1D求取最大值，但前面输入的卷积神经网络是一维的，最终得到一维最大数，
        # 因此可用torch的AdaptiveMaxPool1d(1)达到相同目的
        self.pool2 = nn.AdaptiveMaxPool1d(1)
        self.pool3 = nn.AdaptiveMaxPool1d(1)
        self.pool4 = nn.AdaptiveMaxPool1d(1)
        self.pool5 = nn.AdaptiveMaxPool1d(1)
        self.pool6 = nn.AdaptiveMaxPool1d(1)

        # 全连接层
        self.densecnn = nn.Linear(num_conv_filters * 5, num_conv_filters)
        self.dropoutcnnmid = nn.Dropout(0.5)
        self.dropoutcnn = nn.Dropout(0.5)

        # 输出层
        # 改成多分类标签
        self.output = nn.Linear(num_conv_filters, num_classes)
        pass

    def forward(self, x):
        # 输入
        x = self.embeddingCNN(x)

        # 五层卷积层，这里可能要改改，pytorch希望channel在最后一个维度
        # x2 = F.relu(self.conv2(x))
        # x3 = F.relu(self.conv3(x))
        # x4 = F.relu(self.conv4(x))
        # x5 = F.relu(self.conv5(x))
        # x6 = F.relu(self.conv6(x))
        x2 = F.relu(self.conv2(x.permute(0, 2, 1)))
        x3 = F.relu(self.conv3(x.permute(0, 2, 1)))
        x4 = F.relu(self.conv4(x.permute(0, 2, 1)))
        x5 = F.relu(self.conv5(x.permute(0, 2, 1)))
        x6 = F.relu(self.conv6(x.permute(0, 2, 1)))

        # 池化层
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
            x = torch.cat([x2, x3, x4, x5, x6], dim=1)
            pass
        else:
            # 模态融合,此时张量大小为[batch_size, num_conv_filters]
            x = torch.cat([x2, x3, x4, x5, x6], dim=1)
            pass

        # 全连接层
        x = self.dropoutcnnmid(x)
        x = F.relu(self.densecnn(x))
        x = self.dropoutcnn(x)

        # 激活，输出
        # 多分类输出
        return self.output(x)
        # x = torch.softmax(self.output(x), dim=1)
        # return x
        pass

    pass
