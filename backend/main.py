from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Mishkat Arabic Diacritization API", version="0.1.0")


class DiacritizationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": False}


@app.post("/api/diacritize", status_code=503)
def diacritize(request: DiacritizationRequest):
    return {"detail": "A trained diacritization model is not installed yet."}
