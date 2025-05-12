import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, embedding_dim, max_index, max_string_length):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(max_index, embedding_dim)
        self.extradense = nn.Linear(embedding_dim * max_string_length, 100)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(100, 1)

    def forward(self, input):
        embedded = self.embedding(input)
        flattened = self.flatten(embedded)
        extradense_output = self.extradense(flattened)
        dropout_output = self.dropout(extradense_output)
        output = self.output(dropout_output)
        return torch.sigmoid(output)


class NetMulti(nn.Module):
    """
    num_classes为多分类的标签组
    """

    def __init__(self, embedding_dim, max_index, max_string_length, num_classes):
        super(NetMulti, self).__init__()
        self.embedding = nn.Embedding(max_index, embedding_dim)
        self.extradense = nn.Linear(embedding_dim * max_string_length, 100)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(100, num_classes)
        pass

    def forward(self, input):
        embedded = self.embedding(input)
        flattened = self.flatten(embedded)
        extradense_output = self.extradense(flattened)
        dropout_output = self.dropout(extradense_output)
        output = self.output(dropout_output)
        # 多分类用softmax
        return output
        # return torch.softmax(output, dim=1)

    pass

# 定义模型的输入形状和超参数
# input_shape = 100
# embedding_dim = EMBEDDING_DIMENSION
# max_index = MAX_INDEX
# max_string_length = MAX_STRING_LENGTH

# 创建模型实例
# model = Net(embedding_dim, max_index, max_string_length)
# print(model)
