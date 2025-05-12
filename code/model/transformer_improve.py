import torch
import numpy as np
from timm.layers import trunc_normal_
from torch import nn
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, Mlp
from torch.nn import functional as F


class TopM_MHSA(nn.Module):
    def __init__(self, embed_dim, num_heads, num_mhsa_layers, dim_feedforward, dropout, top_m):
        super().__init__()
        self.nets = nn.ModuleList(
            [MHSA_Block(embed_dim, num_heads, dim_feedforward, dropout, top_m) for _ in range(num_mhsa_layers)])

    def forward(self, x, pos_embed):
        output = x + pos_embed
        for layer in self.nets:
            output = layer(output)
        return output


# TopMAttention 和 MHSA_Block 类保持不变
class TopMAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout, top_m):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.top_m = top_m

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout),
        )
        self.proj_drop = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.zeros(B, self.num_heads, N, N, device=q.device, requires_grad=False)
        index = torch.topk(attn, k=self.top_m, dim=-1, largest=True)[1]
        mask.scatter_(-1, index, 1.)
        attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(x)
        return x


class MHSA_Block(nn.Module):

    def __init__(self, embed_dim, nhead, dim_feedforward, dropout, top_m):
        super().__init__()
        drop_path_rate = 0.1
        self.attn = TopMAttention(embed_dim, nhead, dropout, top_m)
        self.drop_path = DropPath(drop_path_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=dim_feedforward, act_layer=nn.GELU, drop=0.1)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CharEmbedding(nn.Module):
    """字符级嵌入，适用于DGA域名分类"""

    def __init__(self, num_embeddings, embedding_dim):
        super(CharEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        return self.embedding(x)


class Trans_DGA(nn.Module):
    """针对DGA域名识别的模型"""

    def __init__(self, num_classes, vocab_size, embed_dim=256):
        super(Trans_DGA, self).__init__()

        num_heads = 8
        dim_feedforward = embed_dim * 4
        num_mhsa_layers = 1
        dropout = 0.1
        max_len = 255
        top_m = 100

        self.char_embedding = CharEmbedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.topm_mhsa = TopM_MHSA(embed_dim, num_heads, num_mhsa_layers, dim_feedforward, dropout, top_m)
        # self.mlp = nn.Linear(embed_dim, num_classes)
        self.fc_out = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.char_embedding(x)
        x = self.topm_mhsa(x, self.pos_embed)
        x = x.mean(dim=1)
        # x = self.mlp(x)
        x = self.fc_out(x)
        x = torch.sigmoid(x)
        return x


class Trans_DGA_Multi(nn.Module):
    """针对DGA域名识别的模型"""

    def __init__(self, num_classes, vocab_size, embed_dim=256):
        super(Trans_DGA_Multi, self).__init__()
        num_heads = 8
        dim_feedforward = embed_dim * 4
        num_mhsa_layers = 1
        dropout = 0.1
        max_len = 255
        top_m = 100

        self.char_embedding = CharEmbedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.topm_mhsa = TopM_MHSA(embed_dim, num_heads, num_mhsa_layers, dim_feedforward, dropout, top_m)
        self.mlp = nn.Linear(embed_dim, num_classes)
        # self.fc_out = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.char_embedding(x)
        x = self.topm_mhsa(x, self.pos_embed)
        x = x.mean(dim=1)
        # 多分类
        x = self.mlp(x)
        # 单分类
        # x = self.fc_out(x)
        # x = torch.sigmoid(x)
        return x


if __name__ == '__main__':
    vocab_size = 40  # 假设使用ASCII范围作为词汇表大小
    num_classes = 56  # DGA域名识别为二分类问题
    net = Trans_DGA_Multi(num_classes=num_classes, vocab_size=vocab_size)
    x = torch.randint(0, vocab_size, (4, 255))  # 示例输入，假设最大长度为255
    out = net(x)
    print(f"in:{x.shape} --> out:{out.shape}")
