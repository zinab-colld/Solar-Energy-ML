
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "extra_trees_model.joblib"


def load_artifact():
    """Load the trained model and its metadata."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. "
            "Run: python src/train_model.py"
        )

    return joblib.load(MODEL_PATH)


def predict_ac_power(inputs):
    """Estimate AC power from weather and time inputs."""

    artifact = load_artifact()

    features = artifact["features"]
    training_ranges = artifact["training_feature_ranges"]

    missing_features = [
        feature for feature in features
        if feature not in inputs
    ]

    if missing_features:
        raise ValueError(
            f"Missing input features: {missing_features}"
        )

    row = {
        feature: float(inputs[feature])
        for feature in features
    }

    # Basic physical and input validation.
    if row["IRRADIATION"] < 0:
        raise ValueError(
            "Irradiation cannot be negative."
        )

    if row["IRRADIATION"] > 2.0:
        raise ValueError(
            "Irradiation exceeds the configured "
            "validation limit of 2.0."
        )

    for feature in [
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
    ]:
        if not -50 <= row[feature] <= 100:
            raise ValueError(
                f"{feature} must be between -50 and 100 °C."
            )

    if not 0 <= row["HOUR"] <= 23:
        raise ValueError(
            "Hour must be between 0 and 23."
        )

    # Warn when a value is outside the observed training range.
    warnings = []

    for feature in features:
        minimum = training_ranges[feature]["min"]
        maximum = training_ranges[feature]["max"]

        if not minimum <= row[feature] <= maximum:
            warnings.append(
                f"{feature} is outside the training range "
                f"({minimum:.3f} to {maximum:.3f}). "
                "The estimate may be less reliable."
            )

    input_df = pd.DataFrame(
        [row],
        columns=features
    )

    prediction = float(
        artifact["model"].predict(input_df)[0]
    )

    # AC power should not be negative.
    prediction = max(0.0, prediction)

    return {
        "predicted_ac_power": prediction,
        "warnings": warnings,
    }
