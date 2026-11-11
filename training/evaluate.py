"""Evaluate a trusted model checkpoint on held-out JSONL test documents."""
import argparse
import json
from pathlib import Path
from training.inference import Predictor
from training.metrics import score


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--test', type=Path, default=Path('data/processed/test.jsonl'))
    p.add_argument('--output', type=Path, default=Path('artifacts/test_metrics.json'))
    args = p.parse_args()
    predictor = Predictor(args.checkpoint)
    totals = {mode: {'character_errors':0,'characters':0,'word_errors':0,'words':0} for mode in ('all','no_word_end')}
    samples = 0
    with args.test.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            predicted = predictor.predict(item['plain'])
            for mode, ignore in [('all',False),('no_word_end',True)]:
                metrics = score(item['diacritized'], predicted, ignore_word_end=ignore)
                for key in totals[mode]:
                    totals[mode][key] += metrics[key]
            samples += 1
    if samples == 0:
        raise SystemExit('Test split is empty')
    report = {'samples':samples,'metrics':{}}
    for mode, counts in totals.items():
        report['metrics'][mode] = {**counts,
            'der':counts['character_errors']/counts['characters'] if counts['characters'] else None,
            'wer':counts['word_errors']/counts['words'] if counts['words'] else None}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
