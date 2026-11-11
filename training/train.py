"""Train a reproducible BiLSTM baseline on document-split JSONL data."""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from training.model import Diacritizer, PAD, UNK


def load_records(path: Path) -> list[dict]:
    with path.open(encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]
    return [r for r in records if r['plain'] and len(r['plain']) == len(r['labels'])]


class Lines(Dataset):
    def __init__(self, records, chars, tags, max_length):
        self.items = []
        for r in records:
            text, labels = r['plain'], r['labels']
            for start in range(0, len(text), max_length):
                chunk = text[start:start + max_length]
                tag_chunk = labels[start:start + max_length]
                self.items.append(([chars.get(c, 1) for c in chunk], [tags[t] for t in tag_chunk]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(batch):
    lengths = torch.tensor([len(x) for x, _ in batch], dtype=torch.long)
    width = int(lengths.max())
    tokens = torch.zeros((len(batch), width), dtype=torch.long)
    targets = torch.full((len(batch), width), -100, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        tokens[i, :len(x)] = torch.tensor(x)
        targets[i, :len(y)] = torch.tensor(y)
    return tokens, lengths, targets


def evaluate(model, loader, device):
    model.eval()
    errors = total = 0
    with torch.no_grad():
        for x, lengths, y in loader:
            logits = model(x.to(device), lengths)
            predictions = logits.argmax(-1).cpu()
            valid = y != -100
            errors += int(((predictions != y) & valid).sum())
            total += int(valid.sum())
    return errors / total if total else 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('data/processed'))
    p.add_argument('--output', type=Path, default=Path('artifacts/bilstm_best.pt'))
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--max-length', type=int, default=256)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.epochs < 1 or args.max_length < 1 or args.batch_size < 1:
        p.error('epochs, max-length and batch-size must be positive')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train = load_records(args.data / 'train.jsonl')
    val = load_records(args.data / 'val.jsonl')
    if not train or not val:
        raise SystemExit('Training and validation sets must contain labelled examples')
    char_counts = Counter(c for r in train for c in r['plain'])
    chars = {PAD: 0, UNK: 1}
    for c in sorted(char_counts):
        if c not in chars:
            chars[c] = len(chars)
    tags = {'': 0}
    for tag in sorted({t for r in train for t in r['labels']}):
        if tag not in tags:
            tags[tag] = len(tags)
    # Unknown validation tags are not learnable: fail rather than silently relabel.
    unknown = {t for r in val for t in r['labels'] if t not in tags}
    if unknown:
        raise SystemExit(f'Validation includes unseen diacritic labels: {unknown}')
    train_loader = DataLoader(Lines(train, chars, tags, args.max_length), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(Lines(val, chars, tags, args.max_length), batch_size=args.batch_size, collate_fn=collate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = dict(vocab_size=len(chars), tag_count=len(tags))
    model = Diacritizer(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = batches = 0
        for x, lengths, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits.reshape(-1, len(tags)), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach())
            batches += 1
        val_error = evaluate(model, val_loader, device)
        print(f'epoch={epoch} train_loss={running/max(batches,1):.4f} val_tag_error={val_error:.4f}', flush=True)
        if val_error < best:
            best = val_error
            torch.save({'model': model.state_dict(), 'config': config, 'chars': chars, 'tags': tags,
                        'max_length': args.max_length, 'seed': args.seed, 'epoch': epoch,
                        'val_tag_error': best}, args.output)
    print(f'best_val_tag_error={best:.4f} checkpoint={args.output}')


if __name__ == '__main__':
    main()
