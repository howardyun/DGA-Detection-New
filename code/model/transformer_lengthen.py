# 定义Transformer模型
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 定义PositionalEncoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # pe的维度(位置编码最大长度，模型维度)
        pe = torch.zeros(max_len, d_model)
        # 维度为（max_len, 1）：先在[0,max_len]中取max_len个整数，再加一个维度
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 位置编码的除数项：10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        # sin负责奇数；cos负责偶数
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 维度变换：(max_len,d_model)→(1,max_len,d_model)→(max_len,1,d_model)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # 将pe注册为模型缓冲区
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 取pe的前x.size(0)行，即
        # (x.size(0),1,d_model) → (x.size(0),d_model)，拼接到x上
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_layers, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()
        # 创建一个线性变换层，维度input_dim4→d_model
        self.embedding = nn.Embedding(input_dim, d_model)  # 使用嵌入层
        # 生成pe
        self.pos_encoder = PositionalEncoding(d_model, dropout)


        # 生成一层encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                    dropout=dropout)
        # 多层encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)


        # 维度d_model→output_dim
        self.fc = nn.Linear(d_model, output_dim)
        self.d_model = d_model

    def forward(self, src):
        src = src.permute(1, 0)

        # 缩放
        src = self.embedding(src) * np.sqrt(self.d_model)

        # 加上位置嵌入
        src = self.pos_encoder(src)

        output = self.transformer_encoder(src)

        # 调整输出形状为(batch, seq_len, d_model)
        output = output.permute(1, 0, 2)
        # 对所有位置的表示取平均
        output = torch.mean(output, dim=1)
        # 线性变换
        output = self.fc(output)
        # 使用sigmoid激活函数
        output = torch.sigmoid(output)

        return output