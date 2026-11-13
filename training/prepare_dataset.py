"""Stream Tashkeela documents into book-disjoint JSONL splits.

Use --input data/raw/texts.txt, not data/raw (which contains documentation).
This is a capped pilot sampler, not a full-corpus production cleaner.
"""
import argparse
import hashlib
import html
import json
import random
import re
from pathlib import Path

from training.diacritics import split_labels

ARABIC = re.compile(r'[\u0621-\u064a]')
MARK = re.compile(r'[\u064b-\u065f\u0670]')
HTML_TAG = re.compile(r'<[^>]*>')
SPACE = re.compile(r'\s+')
BAD_METADATA = re.compile(r'(?:المكتبة الشاملة|تم إعداد هذا الملف|^الكتاب\s*:|^المؤلف\s*:|https?://|www\.|<\!|\{\\rtf)', re.I)


def clean(line: str) -> str:
    line = html.unescape(line)
    line = HTML_TAG.sub(' ', line)
    return SPACE.sub(' ', line).strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path('data/raw/texts.txt'))
    p.add_argument('--output', type=Path, default=Path('data/processed'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-lines-per-document', type=int, default=300)
    p.add_argument('--max-lines-per-split', type=int, default=12000)
    p.add_argument('--min-arabic-letters', type=int, default=30)
    p.add_argument('--min-mark-ratio', type=float, default=0.65)
    p.add_argument('--max-characters', type=int, default=256)
    args = p.parse_args()
    if args.max_lines_per_document < 1 or args.max_lines_per_split < 1 or args.max_characters < 1:
        p.error('line and character limits must be positive')
    if not 0 <= args.min_mark_ratio <= 2:
        p.error('min-mark-ratio must be between 0 and 2')
    docs = sorted(args.input.rglob('*.txt'))
    if len(docs) < 3:
        raise SystemExit('At least 3 text documents are needed')
    rng = random.Random(args.seed)
    rng.shuffle(docs)
    n = len(docs)
    test_count = max(1, round(n * .1))
    val_count = max(1, round(n * .1))
    splits = {'test':docs[:test_count], 'val':docs[test_count:test_count+val_count],
              'train':docs[test_count+val_count:]}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {'seed':args.seed, 'source':str(args.input), 'splits':{}}
    # Deduplicate exact cleaned lines across all splits. Book assignment is fixed first.
    seen = set()
    for split, documents in splits.items():
        emitted = 0
        chosen = []
        with (args.output / f'{split}.jsonl').open('w', encoding='utf-8') as output:
            for path in documents:
                if emitted >= args.max_lines_per_split:
                    break
                per_doc = 0
                doc_id = hashlib.sha256(str(path.relative_to(args.input)).encode('utf-8')).hexdigest()
                try:
                    source = path.open(encoding='utf-8-sig', errors='replace')
                except OSError as exc:
                    print(f'SKIP {path}: {exc}')
                    continue
                with source:
                    for raw in source:
                        if per_doc >= args.max_lines_per_document or emitted >= args.max_lines_per_split:
                            break
                        if BAD_METADATA.search(raw):
                            continue
                        line = clean(raw)
                        if len(line) > args.max_characters or len(line) < args.min_arabic_letters:
                            continue
                        letters = len(ARABIC.findall(line))
                        if letters < args.min_arabic_letters:
                            continue
                        if len(MARK.findall(line)) / letters < args.min_mark_ratio:
                            continue
                        try:
                            plain, labels = split_labels(line)
                        except ValueError:
                            continue
                        if len(plain) != len(labels) or not any(labels):
                            continue
                        # The model must not learn HTML, isolated orphan marks or duplicate lines.
                        digest = hashlib.sha256(plain.encode('utf-8')).hexdigest()
                        if digest in seen:
                            continue
                        seen.add(digest)
                        output.write(json.dumps({'document_id':doc_id, 'plain':plain,
                            'labels':labels, 'diacritized':line}, ensure_ascii=False) + '\n')
                        per_doc += 1
                        emitted += 1
                if per_doc:
                    chosen.append({'path':str(path.relative_to(args.input)), 'lines':per_doc})
        manifest['splits'][split] = {'lines':emitted, 'documents':chosen}
        print(f'{split}: {len(chosen)} sampled books, {emitted} lines', flush=True)
    (args.output / 'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    if any(manifest['splits'][s]['lines'] == 0 for s in splits):
        raise SystemExit('One or more splits are empty. Adjust filters before training.')
    print('Prepared pilot dataset. Review manifest.json and samples before training.')


if __name__ == '__main__':
    main()
