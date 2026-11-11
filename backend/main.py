"""Mishkat inference API. Model is optional and never silently mocked."""
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Mishkat Arabic Diacritization API", version="0.2.0")


class DiacritizationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


@lru_cache(maxsize=1)
def get_predictor():
    checkpoint = Path(os.environ.get("MISHKAT_MODEL_PATH", "artifacts/bilstm_best.pt"))
    if not checkpoint.is_file():
        return None
    from training.inference import Predictor
    return Predictor(checkpoint)


@app.get("/api/health")
def health():
    checkpoint = Path(os.environ.get("MISHKAT_MODEL_PATH", "artifacts/bilstm_best.pt"))
    return {"status": "ok", "model_loaded": checkpoint.is_file()}


@app.post("/api/diacritize")
def diacritize(request: DiacritizationRequest):
    try:
        predictor = get_predictor()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Model checkpoint could not be loaded") from exc
    if predictor is None:
        raise HTTPException(status_code=503, detail="A trained diacritization model is not installed yet.")
    return {"diacritized_text": predictor.predict(request.text), "model": "contextual-bilstm-baseline"}
