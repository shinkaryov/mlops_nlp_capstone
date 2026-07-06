import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import joblib
import wandb


logger = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    model: Any
    artifact_ref: str
    artifact_dir: str
    classes: List[str]


def load_model_from_wandb() -> ModelBundle:
    artifact_ref = os.environ.get(
        "WANDB_MODEL_ARTIFACT",
        "shinkaryovae-set-university/fanpulse-worldcup-mlops/fanpulse-impression-classifier:production",
    )

    cache_dir = os.environ.get("MODEL_CACHE_DIR", "./model_cache")
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model artifact from W&B: %s", artifact_ref)

    api = wandb.Api()
    artifact = api.artifact(artifact_ref, type="model")

    artifact_dir = artifact.download(root=str(cache_dir_path))
    artifact_dir_path = Path(artifact_dir)

    model_path = artifact_dir_path / "model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"model.joblib was not found inside W&B artifact directory: {artifact_dir_path}"
        )

    model = joblib.load(model_path)

    if not hasattr(model, "predict_proba"):
        raise TypeError("Loaded model does not support predict_proba().")

    classes = [str(c) for c in model.classes_]

    logger.info("Model loaded successfully.")
    logger.info("Artifact ref: %s", artifact_ref)
    logger.info("Artifact dir: %s", artifact_dir)
    logger.info("Classes: %s", classes)

    return ModelBundle(
        model=model,
        artifact_ref=artifact_ref,
        artifact_dir=str(artifact_dir_path),
        classes=classes,
    )