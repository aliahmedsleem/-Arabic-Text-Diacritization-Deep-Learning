"""Contextual character-level Transformer sequence tagger."""
import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_length: int = 8192):
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        scales = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        table = torch.zeros(max_length, dim)
        table[:, 0::2] = torch.sin(positions * scales)
        table[:, 1::2] = torch.cos(positions * scales[:table[:, 1::2].shape[1]])
        self.register_buffer('table', table.unsqueeze(0), persistent=False)

    def forward(self, x):
        if x.size(1) > self.table.size(1):
            raise ValueError('Sequence exceeds positional encoding limit')
        return x + self.table[:, :x.size(1)]


class TransformerDiacritizer(nn.Module):
    def __init__(self, vocab_size: int, tag_count: int, embedding_dim: int = 192,
                 heads: int = 6, layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.position = PositionalEncoding(embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=heads,
            dim_feedforward=embedding_dim * 4, dropout=dropout, batch_first=True,
            activation='gelu', norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, tag_count)
        self.scale = math.sqrt(embedding_dim)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = tokens.eq(0)
        x = self.position(self.embedding(tokens))
        return self.classifier(self.norm(self.encoder(x, src_key_padding_mask=mask)))
