"""FastAPI backend serving predictions on port 11435."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dodag_lm.config import TrainingConfig
from dodag_lm.model import DodagLanguageModel
from dodag_lm.vocab import Vocabulary

logger = logging.getLogger(__name__)
app = FastAPI(title="DoDAG Language Model", version="0.1.0")

MODEL: DodagLanguageModel | None = None
VOCAB: Vocabulary | None = None
CONFIG: TrainingConfig | None = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PredictRequest(BaseModel):
    context: List[str]
    top_k: int = 5


class PredictResponse(BaseModel):
    predictions: List[str]
    scores: List[float]


@app.on_event("startup")
def load_artifacts() -> None:
    global MODEL, VOCAB, CONFIG

    model_path = Path(os.getenv("DODAG_MODEL_PATH", "model.pt"))
    if not model_path.exists():
        logger.warning("Model file %s not found; startup continues without model", model_path)
        return

    payload = torch.load(model_path, map_location=DEVICE)
    tokens = payload["vocab"]
    config_dict = payload["config"]

    VOCAB = Vocabulary()
    VOCAB.idx_to_token = tokens
    VOCAB.token_to_idx = {token: idx for idx, token in enumerate(tokens)}

    CONFIG = TrainingConfig(**config_dict)
    MODEL = DodagLanguageModel(len(VOCAB), CONFIG.embedding_dim, CONFIG.hidden_dim)
    MODEL.load_state_dict(payload["model_state"])
    MODEL.to(DEVICE)
    MODEL.eval()
    logger.info("Model loaded from %s", model_path)


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    return {"status": "ok", "device": DEVICE}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if MODEL is None or VOCAB is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.context:
        parent_token = Vocabulary.root_token
    else:
        parent_token = request.context[-1]

    parent_idx = torch.tensor([VOCAB.encode(parent_token)], device=DEVICE)
    logits = MODEL.predict_child(parent_idx)
    top_k = min(request.top_k, logits.size(-1))
    scores, indices = torch.topk(logits, top_k, dim=-1)

    predictions = [VOCAB.decode(idx.item()) for idx in indices[0]]
    score_list = scores[0].tolist()
    return PredictResponse(predictions=predictions, scores=score_list)


def run() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=11435)


if __name__ == "__main__":
    run()
