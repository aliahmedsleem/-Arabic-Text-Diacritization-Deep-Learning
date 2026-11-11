# System architecture

Arabic RTL web editor → FastAPI inference service → PyTorch contextual sequence tagging model.

## Research stages
1. Curate licensed Arabic diacritized corpora and establish train/validation/test splits by document.
2. Normalize Unicode consistently while preserving original text offsets.
3. Train a reproducible baseline and a contextual Transformer model.
4. Decode diacritics with valid Arabic character/mark constraints.
5. Evaluate DER, WER, and ambiguity cases, with and without final case endings.
6. Integrate a confidence-aware interactive UI, with original and restored text side by side.

## UI direction
Project name: مِشْكَاة. Arabic-first RTL, navy/antique gold/ivory, Amiri for Arabic passages and Cairo for controls. Accessibility and responsive layout are requirements.
