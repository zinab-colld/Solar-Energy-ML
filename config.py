"""Central configuration: paths, feature definitions and validation limits."""

from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = ROOT / "Data"
MODEL_DIR: Path = ROOT / "models"
REPORT_DIR: Path = ROOT / "reports"

RAW_GENERATION_PATH: Path = DATA_DIR / "Plant_1_Generation_Data.csv"
RAW_WEATHER_PATH: Path = DATA_DIR / "Plant_1_Weather_Sensor_Data.csv"
MERGED_PATH: Path = DATA_DIR / "solar_plant1_merged.csv"

MODEL_PATH: Path = MODEL_DIR / "extra_trees_model.joblib"
METADATA_PATH: Path = MODEL_DIR / "model_metadata.json"
PREDICTIONS_PATH: Path = REPORT_DIR / "test_predictions.csv"
VALIDATION_REPORT_PATH: Path = REPORT_DIR / "data_validation.json"

# Inputs a user supplies, and the full engineered set the model is trained on.
BASE_FEATURES: list[str] = [
    "IRRADIATION",
    "MODULE_TEMPERATURE",
    "AMBIENT_TEMPERATURE",
    "HOUR",
]
MODEL_FEATURES: list[str] = BASE_FEATURES + ["TEMP_DELTA", "HOUR_SIN", "HOUR_COS"]
TARGET: str = "AC_POWER"

# Bump when the feature set changes so stale model files are rejected.
FEATURE_VERSION: int = 2

EXPECTED_INVERTERS: int = 22
PARTIAL_IRRADIATION_THRESHOLD: float = 0.3

MAX_IRRADIATION: float = 2.0
TEMP_LIMITS: tuple[float, float] = (-50.0, 100.0)

TEST_FRACTION: float = 0.2
MIN_TRAINING_DAYS: int = 10
RANDOM_STATE: int = 42