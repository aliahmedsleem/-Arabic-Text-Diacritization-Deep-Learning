"""Prepare licensed diacritized text into JSONL with document-level splits.

Input: UTF-8 .txt files, one document per file, in --input directory.
No dataset is downloaded automatically; verify source licenses first.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path
from training.diacritics import split_labels


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('data/processed'))
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    documents = sorted(args.input.rglob('*.txt'))
    if not documents:
        raise SystemExit('No .txt documents found')
    random.Random(args.seed).shuffle(documents)
    n = len(documents)
    if n < 3:
        raise SystemExit('At least 3 independent documents are required for train/val/test splits')
    val_count = max(1, round(n * .1))
    test_count = max(1, round(n * .1))
    split_docs = {'test': documents[:test_count], 'val': documents[test_count:test_count+val_count], 'train': documents[test_count+val_count:]}
    args.output.mkdir(parents=True, exist_ok=True)
    for split, docs in split_docs.items():
        count = 0
        with (args.output / f'{split}.jsonl').open('w', encoding='utf-8') as output:
            for path in docs:
                doc_id = hashlib.sha256(path.read_bytes()).hexdigest()
                for line in path.read_text(encoding='utf-8-sig').splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        plain, labels = split_labels(line)
                    except ValueError:
                        continue
                    if not any(labels):
                        continue
                    output.write(json.dumps({'document_id':doc_id,'plain':plain,'labels':labels,'diacritized':line},ensure_ascii=False)+'\n')
                    count += 1
        print(f'{split}: {len(docs)} documents, {count} lines')


if __name__ == '__main__':
    main()
