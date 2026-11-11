# مِشْكَاة — دليل التدريب الأولي

## 1. Requirements

Python 3.11+ is recommended. Install a PyTorch build compatible with your CPU/CUDA first (see official PyTorch installation instructions), then:

```bash
pip install -r training/requirements.txt
pip install -r backend/requirements-dev.txt
python -m pytest training/tests backend/tests -q
```

## 2. Dataset

Obtain a diacritized Arabic corpus under a license compatible with your use. Verify its provenance and document boundaries. Place each independent document as a UTF-8 .txt file under `data/raw/`.

```bash
python -m training.prepare_dataset --input data/raw --output data/processed --seed 42
```

Inspect JSONL outputs before training; check duplicates, markup, non-Arabic characters, Unicode variants and label distribution. A document split is not a substitute for near-duplicate removal.

## 3. Baseline training

```bash
python -m training.train --data data/processed --epochs 12 --batch-size 32 --max-length 256 --output artifacts/bilstm_best.pt
```

On CUDA-enabled systems PyTorch automatically uses the GPU. This baseline optimizes token-label cross entropy. Its `val_tag_error` includes whitespace and punctuation, so **do not** report it as Arabic DER. Use the independent evaluation metrics in `training/metrics.py` on held-out test predictions.

## 4. Run the interface

```bash
uvicorn backend.main:app --reload
# in another terminal:
cd frontend && npm install && npm run dev
```

Once a trusted checkpoint is saved at `artifacts/bilstm_best.pt`, restart the backend. Alternatively set `MISHKAT_MODEL_PATH` to the checkpoint path. Before training, inference returns HTTP 503 by design.

## Limitations

This is a research baseline, not a verified production diacritizer. Fixed-length chunking can cut linguistic context. Additional evaluation, mark validity constraints, model comparisons and corpus licensing review are required.
