from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1)
    candidate_player: str = Field(..., min_length=1)
    position: Optional[str] = ""
    likes: Optional[float] = 0.0
    retweets: Optional[float] = 0.0


class PredictionResponse(BaseModel):
    candidate_player: str
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    aboutness: float
    sentiment_signal: float
    model_version: str
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    examples: List[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    model_version: str
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


class ModelInfoResponse(BaseModel):
    model_version: str
    artifact_ref: str
    classes: List[str]
    model_cache_dir: str