import logging
import time
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.inference.model_loader import ModelBundle, load_model_from_wandb
from src.inference.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="FanPulse World Cup Inference API",
    description="Player-level fan impression classifier loaded from W&B Model Registry.",
    version="1.0.0",
)

MODEL_BUNDLE: ModelBundle | None = None


def make_model_input(request: PredictionRequest) -> str:
    return (
        f"candidate_player: {request.candidate_player or ''} "
        f"position: {request.position or ''} "
        f"tweet: {request.text or ''}"
    )


def predict_one(request: PredictionRequest, model_bundle: ModelBundle) -> PredictionResponse:
    start = time.perf_counter()

    model_input = make_model_input(request)
    X = pd.Series([model_input])

    proba = model_bundle.model.predict_proba(X)[0]
    classes = model_bundle.classes

    probabilities = {
        cls: float(prob)
        for cls, prob in zip(classes, proba)
    }

    prediction = max(probabilities, key=probabilities.get)
    confidence = probabilities[prediction]

    p_positive = probabilities.get("positive", 0.0)
    p_negative = probabilities.get("negative", 0.0)
    p_not_about = probabilities.get("not_about_player", 0.0)

    aboutness = 1.0 - p_not_about
    sentiment_signal = p_positive - p_negative

    latency_ms = (time.perf_counter() - start) * 1000.0

    return PredictionResponse(
        candidate_player=request.candidate_player,
        prediction=prediction,
        confidence=float(confidence),
        probabilities=probabilities,
        aboutness=float(aboutness),
        sentiment_signal=float(sentiment_signal),
        model_version=model_bundle.artifact_ref,
        latency_ms=float(latency_ms),
    )


@app.on_event("startup")
def startup_event():
    global MODEL_BUNDLE

    try:
        MODEL_BUNDLE = load_model_from_wandb()
    except Exception:
        logger.exception("Failed to load model from W&B.")
        raise


@app.get("/", response_model=HealthResponse)
def root():
    return health()


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if MODEL_BUNDLE is not None else "model_not_loaded",
        model_loaded=MODEL_BUNDLE is not None,
        model_version=MODEL_BUNDLE.artifact_ref if MODEL_BUNDLE else "",
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    if MODEL_BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    return ModelInfoResponse(
        model_version=MODEL_BUNDLE.artifact_ref,
        artifact_ref=MODEL_BUNDLE.artifact_ref,
        classes=MODEL_BUNDLE.classes,
        model_cache_dir=MODEL_BUNDLE.artifact_dir,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if MODEL_BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    return predict_one(request, MODEL_BUNDLE)


@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    if MODEL_BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    start = time.perf_counter()

    predictions: List[PredictionResponse] = [
        predict_one(example, MODEL_BUNDLE)
        for example in request.examples
    ]

    total_latency_ms = (time.perf_counter() - start) * 1000.0

    return BatchPredictionResponse(
        predictions=predictions,
        model_version=MODEL_BUNDLE.artifact_ref,
        total_latency_ms=float(total_latency_ms),
    )