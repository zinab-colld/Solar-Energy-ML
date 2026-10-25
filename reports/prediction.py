"""Model loading and safe inference."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

import config
from reports.src.data_processing import engineer_features

REQUIRED_ARTIFACT_KEYS = {
    "model",
    "features",
    "training_feature_ranges",
    "target_range",
    "feature_version",
}


class ModelNotAvailableError(FileNotFoundError):
    """The trained model file does not exist or is outdated."""


class InputValidationError(ValueError):
    """The supplied inputs are invalid or physically implausible."""


@dataclass(frozen=True)
class PredictionResult:
    predicted_ac_power: float
    inputs: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    in_training_range: bool = True


@lru_cache(maxsize=2)
def _read_artifact(path: str, mtime_ns: int) -> dict[str, Any]:
    # mtime_ns is part of the cache key, so retraining refreshes the cache.
    return joblib.load(path)


def load_artifact(path: Path = config.MODEL_PATH) -> dict[str, Any]:
    """Load the trained model artifact (cached, refreshed when the file changes)."""
    if not path.exists():
        raise ModelNotAvailableError(
            "Trained model not found. Train it with: python -m reports.train_model"
        )
    artifact = _read_artifact(str(path), path.stat().st_mtime_ns)
    if (
        not REQUIRED_ARTIFACT_KEYS.issubset(artifact)
        or artifact["feature_version"] != config.FEATURE_VERSION
    ):
        raise ModelNotAvailableError(
            "The saved model is outdated. Retrain it with: python -m reports.train_model"
        )
    return artifact


def _coerce_inputs(inputs: Mapping[str, Any]) -> dict[str, float]:
    missing = [f for f in config.BASE_FEATURES if f not in inputs]
    if missing:
        raise InputValidationError(f"Missing input features: {missing}")

    row: dict[str, float] = {}
    for feature in config.BASE_FEATURES:
        try:
            value = float(inputs[feature])
        except (TypeError, ValueError) as error:
            raise InputValidationError(f"{feature} must be a number.") from error
        if not math.isfinite(value):
            raise InputValidationError(f"{feature} must be a finite number.")
        row[feature] = value
    return row


def _validate_physical_limits(row: Mapping[str, float]) -> None:
    if row["IRRADIATION"] < 0:
        raise InputValidationError("Irradiation cannot be negative.")
    if row["IRRADIATION"] > config.MAX_IRRADIATION:
        raise InputValidationError(
            f"Irradiation exceeds the validation limit of {config.MAX_IRRADIATION}."
        )
    low, high = config.TEMP_LIMITS
    for feature in ("AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE"):
        if not low <= row[feature] <= high:
            raise InputValidationError(
                f"{feature} must be between {low:.0f} and {high:.0f} °C."
            )
    if not 0 <= row["HOUR"] <= 23 or row["HOUR"] != int(row["HOUR"]):
        raise InputValidationError("Hour must be a whole number from 0 to 23.")


def _collect_warnings(
    row: Mapping[str, float], ranges: Mapping[str, Mapping[str, float]]
) -> tuple[list[str], bool]:
    warnings: list[str] = []
    in_range = True

    for feature in config.BASE_FEATURES:
        low, high = ranges[feature]["min"], ranges[feature]["max"]
        if not low <= row[feature] <= high:
            in_range = False
            warnings.append(
                f"{feature.replace('_', ' ').title()} is outside the training range "
                f"({low:.2f} to {high:.2f}). The estimate may be less reliable."
            )

    irradiation, hour = row["IRRADIATION"], row["HOUR"]
    if irradiation > 0.1 and (hour < 5 or hour > 19):
        warnings.append(
            "Meaningful irradiation at this hour is unusual; check the hour and sensor values."
        )
    if irradiation < 0.02 and 8 <= hour <= 16:
        warnings.append(
            "Near-zero irradiation at midday suggests heavy cloud cover or a sensor fault."
        )
    if irradiation > 0.1 and row["MODULE_TEMPERATURE"] < row["AMBIENT_TEMPERATURE"]:
        warnings.append(
            "Module temperature is below ambient in daylight, which is physically unusual."
        )
    return warnings, in_range


def predict_batch(
    frame: pd.DataFrame, artifact: Mapping[str, Any] | None = None
) -> np.ndarray:
    """Predict for many rows of already-validated base features (clipped at 0)."""
    artifact = artifact or load_artifact()
    engineered = engineer_features(frame[config.BASE_FEATURES].astype(float))
    raw = artifact["model"].predict(engineered[artifact["features"]])
    return np.clip(raw, 0.0, None)


def predict_ac_power(inputs: Mapping[str, Any]) -> PredictionResult:
    """Validate inputs, estimate AC power and report reliability warnings."""
    artifact = load_artifact()
    row = _coerce_inputs(inputs)
    _validate_physical_limits(row)
    warnings, in_range = _collect_warnings(row, artifact["training_feature_ranges"])

    estimate = float(predict_batch(pd.DataFrame([row]), artifact)[0])
    return PredictionResult(
        predicted_ac_power=estimate,
        inputs=row,
        warnings=warnings,
        in_training_range=in_range,
    )