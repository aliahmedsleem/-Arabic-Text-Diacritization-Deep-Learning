"""Character-label DER and word-level WER for Arabic diacritization.

Scores include punctuation/whitespace only as word boundaries; DER denominator
is Arabic base characters, with optional omission of each word's final mark.
"""
import re
from training.diacritics import ARABIC_BASE_RE, split_labels


def score(gold: str, predicted: str, ignore_word_end: bool = False) -> dict:
    gold_base, gold_labels = split_labels(gold)
    pred_base, pred_labels = split_labels(predicted)
    if gold_base != pred_base:
        raise ValueError('Prediction and reference must have identical undiacritized text')
    indices = [i for i,c in enumerate(gold_base) if ARABIC_BASE_RE.fullmatch(c)]
    if ignore_word_end:
        indices = [i for i in indices if i+1 < len(gold_base) and ARABIC_BASE_RE.fullmatch(gold_base[i+1])]
    errors = sum(gold_labels[i] != pred_labels[i] for i in indices)
    # Word boundaries are whitespace and punctuation. Count only Arabic-containing tokens.
    spans = list(re.finditer(r'[\u0621-\u064a\u066e-\u066f\u0671-\u06d3]+', gold_base))
    word_errors = 0
    for span in spans:
        positions = [i for i in range(span.start(),span.end()) if i in indices]
        if positions and any(gold_labels[i] != pred_labels[i] for i in positions):
            word_errors += 1
    return {'der': errors/len(indices) if indices else 0.0,
            'wer': word_errors/len(spans) if spans else 0.0,
            'character_errors': errors,'characters':len(indices),
            'word_errors':word_errors,'words':len(spans)}
