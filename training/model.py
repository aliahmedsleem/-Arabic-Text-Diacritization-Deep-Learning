"""Character-level contextual BiLSTM tagger for Arabic diacritization."""
import torch
from torch import nn

PAD = '<PAD>'
UNK = '<UNK>'


class Diacritizer(nn.Module):
    def __init__(self, vocab_size: int, tag_count: int, embedding_dim: int = 128, hidden_dim: int = 192, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers=layers, batch_first=True,
                               bidirectional=True, dropout=dropout if layers > 1 else 0)
        self.classifier = nn.Linear(hidden_dim * 2, tag_count)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.encoder(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True, total_length=tokens.shape[1])
        return self.classifier(unpacked)
