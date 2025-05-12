import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]


class TopMAttention(nn.Module):
    def __init__(self, embed_size, heads, m_value):
        super(TopMAttention, self).__init__()
        self.heads = heads
        self.m_value = m_value
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size, "Embed size must be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, value, key, query):
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        # Split the embedding into multiple heads
        values = self.values(value).view(N, value_len, self.heads, self.head_dim)
        keys = self.keys(key).view(N, key_len, self.heads, self.head_dim)
        queries = self.queries(query).view(N, query_len, self.heads, self.head_dim)

        # Calculate the dot product for similarity scores
        energy = torch.einsum('nqhd,nkhd->nhqk', [queries, keys])

        # Apply the Top-M selection
        top_m_energy, top_m_indices = torch.topk(energy, self.m_value, dim=-1)
        attention = F.softmax(top_m_energy, dim=-1)

        # Instead of using expand and gather, directly use indexing which is more straightforward
        # Create masks based on top_m_indices and use them to filter values
        N, H, Q, M = top_m_indices.size()
        _, _, _, D = values.size()

        # Rearrange values and top_m_indices for direct indexing
        values = values.permute(0, 2, 1, 3).contiguous().view(N, H, -1, D)  # Shape: [N, H, value_len * head_dim, D]
        top_m_indices_flat = top_m_indices.view(N, H, -1)  # Flatten the top_m_indices for direct indexing
        selected_values = torch.gather(values, 2, top_m_indices_flat.unsqueeze(-1).expand(-1, -1, -1, D))

        # Reapply the dimension to selected_values to match attention weights dimension
        selected_values = selected_values.view(N, H, Q, M, D)

        # Final weighted sum
        out = torch.einsum('nhqm,nhqmd->nhqd', [attention, selected_values])
        out = out.reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, m_value, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = TopMAttention(embed_size, heads, m_value)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)
        x = self.norm1(attention + query)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out


class DGAAttentionModel(nn.Module):
    def __init__(self, embed_size, heads, m_value, num_classes, forward_expansion, num_layers, max_length):
        super(DGAAttentionModel, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(max_length, embed_size)
        self.position_encoding = PositionalEncoding(embed_size, max_length)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, m_value, forward_expansion)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        N, seq_length = x.size()
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out)

        out = out.mean(dim=1)
        out = self.fc_out(out)
        out = out.squeeze(1)

        return out
