# Dataset acquisition and evaluation protocol

## Candidate corpus

Tashkeela is a commonly used corpus of diacritized Arabic texts. Verify the original distribution, license and attribution requirements before downloading or publishing any data. This project does not bundle third-party corpora.

## Preparation checklist

1. Record source URL, download date, version, license and attribution.
2. Remove HTML/markup and identify independent source documents.
3. Preserve raw source files separately and record SHA-256 hashes.
4. Deduplicate near-identical documents **before** splitting.
5. Normalize Unicode consistently and inspect ambiguous combining-mark order.
6. Split by document with fixed seed; never use the held-out test set for model selection.
7. Inspect the distribution of label combinations, punctuation and case endings.
8. Report DER and WER with and without word-final diacritics, plus examples of common errors.

## Model comparison

Compare the BiLSTM baseline against a Transformer under equivalent data splits and preprocessing. Record training time, hardware, seed, number of parameters, epochs and validation selection criteria. Do not claim a model wins without held-out test results.
