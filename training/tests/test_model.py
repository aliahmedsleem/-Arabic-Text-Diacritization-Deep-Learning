import torch
from training.model import Diacritizer


def test_bilstm_shapes():
    model = Diacritizer(vocab_size=20, tag_count=8)
    tokens = torch.tensor([[2, 3, 0], [4, 5, 6]])
    lengths = torch.tensor([2, 3])
    assert model(tokens, lengths).shape == (2, 3, 8)
