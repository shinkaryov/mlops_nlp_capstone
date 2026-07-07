import logging
import time
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

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


# -------------------------
# Prometheus metrics
# -------------------------

HTTP_REQUESTS_TOTAL = Counter(
    "fanpulse_http_requests_total",
    "Total HTTP requests.",
    ["method", "endpoint", "status_code"],
)

HTTP_REQUEST_LATENCY_SECONDS = Histogram(
    "fanpulse_http_request_latency_seconds",
    "HTTP request latency in seconds.",
    ["method", "endpoint"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

PREDICTIONS_TOTAL = Counter(
    "fanpulse_predictions_total",
    "Total number of model predictions.",
    ["prediction"],
)

PREDICTION_LATENCY_SECONDS = Histogram(
    "fanpulse_prediction_latency_seconds",
    "Model prediction latency in seconds.",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

PREDICTION_ERRORS_TOTAL = Counter(
    "fanpulse_prediction_errors_total",
    "Total number of prediction errors.",
)

MODEL_LOADED = Gauge(
    "fanpulse_model_loaded",
    "Whether the model is currently loaded. 1 = loaded, 0 = not loaded.",
)

MODEL_INFO = Gauge(
    "fanpulse_model_info",
    "Model registry metadata. Value is always 1 when model is loaded.",
    ["model_version", "artifact_ref"],
)


# -------------------------
# FastAPI app
# -------------------------

app = FastAPI(
    title="FanPulse World Cup Inference API",
    description="Player-level fan impression classifier loaded from W&B Model Registry.",
    version="1.0.0",
)

MODEL_BUNDLE: ModelBundle | None = None


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    endpoint = request.url.path
    status_code = "500"

    try:
        response = await call_next(request)
        status_code = str(response.status_code)
        return response
    finally:
        latency = time.perf_counter() - start

        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=status_code,
        ).inc()

        HTTP_REQUEST_LATENCY_SECONDS.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe(latency)


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

    latency_seconds = time.perf_counter() - start
    latency_ms = latency_seconds * 1000.0

    PREDICTIONS_TOTAL.labels(prediction=prediction).inc()
    PREDICTION_LATENCY_SECONDS.observe(latency_seconds)

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

    MODEL_LOADED.set(0)

    try:
        MODEL_BUNDLE = load_model_from_wandb()

        MODEL_LOADED.set(1)
        MODEL_INFO.labels(
            model_version=MODEL_BUNDLE.artifact_ref,
            artifact_ref=MODEL_BUNDLE.artifact_ref,
        ).set(1)

    except Exception:
        MODEL_LOADED.set(0)
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


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if MODEL_BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        return predict_one(request, MODEL_BUNDLE)
    except Exception:
        PREDICTION_ERRORS_TOTAL.inc()
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail="Prediction failed.")


@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    if MODEL_BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    start = time.perf_counter()

    try:
        predictions: List[PredictionResponse] = [
            predict_one(example, MODEL_BUNDLE)
            for example in request.examples
        ]
    except Exception:
        PREDICTION_ERRORS_TOTAL.inc()
        logger.exception("Batch prediction failed.")
        raise HTTPException(status_code=500, detail="Batch prediction failed.")

    total_latency_ms = (time.perf_counter() - start) * 1000.0

    return BatchPredictionResponse(
        predictions=predictions,
        model_version=MODEL_BUNDLE.artifact_ref,
        total_latency_ms=float(total_latency_ms),
    )