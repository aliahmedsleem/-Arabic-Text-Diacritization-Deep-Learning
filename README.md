# مِشْكَاة | Arabic Contextual Diacritization

نظام بحثي لتشكيل النصوص العربية واستعادة الحركات اعتماداً على السياق باستخدام التعلم العميق.

## Architecture
- `frontend/`: Arabic-first RTL React interface (planned)
- `backend/`: FastAPI inference API
- `training/`: PyTorch experiments and model evaluation (planned)

> **Research status:** Initial scaffold. The demo API does not contain a trained model. Do not report its output as AI predictions.

## Local API
```bash
python -m venv .venv
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

API documentation: http://127.0.0.1:8000/docs

## Evaluation plan
Compare character-level DER and word-level WER on a held-out, leakage-free dataset. Record results with and without case-ending diacritics.
