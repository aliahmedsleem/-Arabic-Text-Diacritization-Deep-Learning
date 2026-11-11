import torch
from training.transformer import TransformerDiacritizer


def test_transformer_output_shape():
    model = TransformerDiacritizer(vocab_size=24, tag_count=10, embedding_dim=48, heads=4, layers=1)
    x = torch.tensor([[2,3,0],[4,5,6]])
    lengths = torch.tensor([2,3])
    assert model(x,lengths).shape == (2,3,10)
