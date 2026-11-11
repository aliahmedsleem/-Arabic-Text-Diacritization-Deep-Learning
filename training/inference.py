"""Load a trusted locally trained checkpoint and diacritize text."""
from pathlib import Path
import torch
from training.diacritics import apply_labels, strip_diacritics
from training.model import Diacritizer
from training.transformer import TransformerDiacritizer


class Predictor:
    def __init__(self, checkpoint: str | Path, device: str = 'cpu'):
        self.device = torch.device(device)
        # Only load trusted locally generated checkpoint files.
        payload = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.chars = payload['chars']
        self.tags = {value: key for key, value in payload['tags'].items()}
        self.max_length = payload['max_length']
        architecture = payload.get('architecture', 'bilstm')
        if architecture not in ('bilstm', 'transformer'):
            raise ValueError(f'Unsupported architecture: {architecture}')
        model_class = TransformerDiacritizer if architecture == 'transformer' else Diacritizer
        self.model = model_class(**payload['config']).to(self.device)
        self.model.load_state_dict(payload['model'])
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text: str) -> str:
        plain = strip_diacritics(text)
        if not plain:
            return plain
        labels = []
        for start in range(0, len(plain), self.max_length):
            segment = plain[start:start + self.max_length]
            tokens = torch.tensor([[self.chars.get(c, 1) for c in segment]], device=self.device)
            lengths = torch.tensor([len(segment)])
            prediction = self.model(tokens, lengths).argmax(-1)[0].tolist()
            labels.extend(self.tags[i] for i in prediction)
        return apply_labels(plain, labels)
