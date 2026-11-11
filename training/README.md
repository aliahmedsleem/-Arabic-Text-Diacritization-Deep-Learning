# Arabic diacritization — research pipeline

## Stage 1: reproducible data preparation and evaluation

The repository now includes Unicode normalization, character-level mark extraction,
reconstruction, document-level dataset splits, and DER/WER evaluation.

### Prepare data

Place **licensed** diacritized UTF-8 documents in `data/raw/`, one document per `.txt` file.
Do not upload raw datasets without permission.

```bash
python -m training.prepare_dataset --input data/raw --output data/processed --seed 42
python -m pytest training/tests -q
```

The resulting train/val/test files contain JSONL records with original text, unmarked text,
character-aligned diacritic labels and a source-document content hash.

### Important research constraints

- Splits are by input document, not by sentence. Near-duplicate documents still require deduplication.
- Dataset licenses, provenance and coverage must be recorded before training.
- Character-level DER and word-level WER are reported both with and without word-final diacritics.
- The current code is **preprocessing and evaluation only**; no model has been trained.
- In the next stage, fit a baseline and contextual sequence model with validation-only hyperparameter selection.
