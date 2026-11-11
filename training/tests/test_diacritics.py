from training.diacritics import split_labels, apply_labels, strip_diacritics
from training.metrics import score


def test_roundtrip():
    text = 'الْعِلْمُ نُورٌ'
    base, labels = split_labels(text)
    assert base == 'العلم نور'
    assert apply_labels(base, labels) == text
    assert strip_diacritics(text) == base


def test_identical_prediction():
    result = score('كَتَبَ', 'كَتَبَ')
    assert result['der'] == 0 and result['wer'] == 0


def test_mismatch():
    result = score('كَتَبَ', 'كُتِبَ')
    assert result['character_errors'] == 2
    assert result['word_errors'] == 1


def test_ignore_case_ending():
    assert score('كِتَابٌ', 'كِتَابُ', ignore_word_end=True)['der'] == 0
