"""Arabic diacritic vocabulary and Unicode helpers."""
import re
import unicodedata

MARKS = tuple(chr(i) for i in range(0x64B, 0x653)) + ('\u0670',)
# Diacritic tags represent zero or more marks attached to each base character.
DIACRITIC_RE = re.compile(r'[\u064b-\u065f\u0670\u06d6-\u06ed]')
ARABIC_BASE_RE = re.compile(r'[\u0621-\u064a\u066e-\u066f\u0671-\u06d3]')


def normalize(text: str) -> str:
    return unicodedata.normalize('NFC', text)


def strip_diacritics(text: str) -> str:
    return DIACRITIC_RE.sub('', normalize(text))


def split_labels(text: str) -> tuple[str, list[str]]:
    """Extract per-base-character mark labels; raise for orphan marks.

    NFC may compose some Arabic combinations; output base is normalized text
    with diacritics removed. Non-Arabic characters receive empty labels.
    """
    text = normalize(text)
    bases: list[str] = []
    labels: list[str] = []
    for char in text:
        if DIACRITIC_RE.fullmatch(char):
            if not bases or not ARABIC_BASE_RE.fullmatch(bases[-1]):
                raise ValueError('Orphan or unsupported Arabic diacritic')
            labels[-1] += char
        else:
            bases.append(char)
            labels.append('')
    return ''.join(bases), labels


def apply_labels(base: str, labels: list[str]) -> str:
    if len(base) != len(labels):
        raise ValueError('Label count must match base text length')
    return ''.join(c + label for c, label in zip(base, labels))
