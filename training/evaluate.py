"""Evaluate a trained model on validation by default; reserve test for final reporting."""
import argparse
import json
from pathlib import Path

from training.diacritics import strip_diacritics
from training.inference import Predictor
from training.metrics import score


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--input', type=Path, default=Path('data/processed/val.jsonl'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = p.parse_args()
    predictor = Predictor(args.checkpoint, device=args.device)
    totals = {mode: {'character_errors': 0, 'characters': 0, 'word_errors': 0, 'words': 0}
              for mode in ('all', 'no_word_end')}
    samples = 0
    with args.input.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            predicted = predictor.predict(item['plain'])
            if strip_diacritics(predicted) != item['plain']:
                raise ValueError(f'Base text changed at sample {samples + 1}')
            for mode, ignore in [('all', False), ('no_word_end', True)]:
                metrics = score(item['diacritized'], predicted, ignore_word_end=ignore)
                for key in totals[mode]:
                    totals[mode][key] += metrics[key]
            samples += 1
            if samples % 500 == 0:
                print(f'Evaluated {samples} samples', flush=True)
    if samples == 0:
        raise SystemExit('Evaluation input is empty')
    report = {'checkpoint': str(args.checkpoint), 'input': str(args.input),
              'samples': samples, 'metrics': {}}
    for mode, counts in totals.items():
        report['metrics'][mode] = {**counts,
            'der': counts['character_errors'] / counts['characters'] if counts['characters'] else None,
            'wer': counts['word_errors'] / counts['words'] if counts['words'] else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
