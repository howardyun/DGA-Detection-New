import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


class DGAClassifier(nn.Module):
    def __init__(self, input_vocab_size, embed_size, num_heads, num_encoder_layers, num_classes, max_len=255):
        super(DGAClassifier, self).__init__()
        self.embed_size = embed_size
        self.max_len = max_len
        self.embedding = nn.Embedding(input_vocab_size, embed_size)
        self.positional_encoding = self._generate_positional_encoding(max_len, embed_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.fc_out = nn.Linear(embed_size, num_classes)

    def _generate_positional_encoding(self, max_len, embed_size):
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size))
        positional_encoding = torch.zeros(max_len, embed_size)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        return positional_encoding

    # 原始版本
    def forward(self, x):
        embeddings = self.embedding(x) + self.positional_encoding[:x.size(1), :].to(x.device)
        transformer_output = self.transformer_encoder(embeddings.permute(1, 0, 2))
        # Assume the first token's output to be the representation of the sequence
        out = self.fc_out(transformer_output[0])
        return out



# Example of using the model
# input_vocab_size = 40  # Assuming ASCII + some special tokens
# embed_size = 512
# num_heads = 8
# num_encoder_layers = 3
# num_classes = 2  # Binary classification


import math
import torch
import torch.nn as nn

import math
import torch
import torch.nn as nn

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# class DGAClassifier(nn.Module):
#     def __init__(self, input_vocab_size, embed_size, num_heads, num_encoder_layers, num_classes, max_len=255):
#         super(DGAClassifier, self).__init__()
#         self.embed_size = embed_size
#         self.max_len = max_len
#         self.embedding = nn.Embedding(input_vocab_size, embed_size)
#         self.positional_encoding = self._generate_positional_encoding(max_len, embed_size)
#
#         encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
#
#         self.fc_out = nn.Linear(embed_size, num_classes)
#
#     def _generate_positional_encoding(self, max_len, embed_size):
#         position = torch.arange(0, max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size))
#         positional_encoding = torch.zeros(max_len, embed_size)
#         positional_encoding[:, 0::2] = torch.sin(position * div_term)
#         positional_encoding[:, 1::2] = torch.cos(position * div_term)
#         return positional_encoding
#
#     def forward(self, x):
#         # 生成掩码
#         mask = self.generate_padding_mask(x)
#
#         embeddings = self.embedding(x) + self.positional_encoding[:x.size(1), :].to(x.device)
#
#         transformer_output = self.transformer_encoder(embeddings.transpose(0, 1),
#                                                       src_key_padding_mask=mask.transpose(0, 1))  # 使用transpose调整掩码的形状
#
#         # Assume the first token's output to be the representation of the sequence
#         out = self.fc_out(transformer_output.mean(dim=0))  # 对序列维度求均值，得到单个向量表示
#         return out
#
#     def generate_padding_mask(self, x):
#         """
#         生成填充掩码
#         Args:
#             x: 输入序列张量，shape为(batch_size, seq_len)
#
#         Returns:
#             mask: 填充掩码张量，shape为(batch_size, seq_len)
#         """
#         mask = (x == 0)  # 找到填充位置，值为True
#         return mask



