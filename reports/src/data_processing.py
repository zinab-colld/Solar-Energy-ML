"""Data loading, validation and feature engineering."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # allow `python reports/src/data_processing.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config

FEATURES: list[str] = config.MODEL_FEATURES
BASE_FEATURES: list[str] = config.BASE_FEATURES
TARGET: str = config.TARGET

REQUIRED_COLUMNS = [
    "DATE_TIME",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION",
    "AC_POWER",
]
MEASUREMENT_COLUMNS = [
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION",
    "AC_POWER",
]


class DataValidationError(ValueError):
    """Raised when the dataset cannot be used safely."""


def parse_datetime(series: pd.Series) -> pd.Series:
    """Parse timestamps from either `dd-mm-YYYY HH:MM` (Kaggle raw generation
    file) or ISO format (weather file and merged file) without day/month mix-ups.
    """
    text = series.astype("string").str.strip()
    parsed = pd.to_datetime(text, format="%d-%m-%Y %H:%M", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        fallback = pd.to_datetime(text.where(missing), errors="coerce")
        parsed = parsed.fillna(fallback)
    return parsed


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features. Shared by training and inference to avoid skew."""
    missing = [c for c in BASE_FEATURES if c not in df.columns]
    if missing:
        raise DataValidationError(f"Cannot engineer features, missing: {missing}")

    out = df.copy()
    out["TEMP_DELTA"] = out["MODULE_TEMPERATURE"] - out["AMBIENT_TEMPERATURE"]
    angle = 2.0 * np.pi * out["HOUR"].astype(float) / 24.0
    out["HOUR_SIN"] = np.sin(angle)
    out["HOUR_COS"] = np.cos(angle)
    return out


def _inverter_counts_from_raw(path: Path) -> pd.DataFrame:
    generation = pd.read_csv(path)
    generation.columns = generation.columns.str.strip()
    if not {"DATE_TIME", "SOURCE_KEY"}.issubset(generation.columns):
        raise DataValidationError(
            "The raw generation file must contain DATE_TIME and SOURCE_KEY."
        )
    generation["DATE_TIME"] = parse_datetime(generation["DATE_TIME"])
    generation = generation.dropna(subset=["DATE_TIME"])
    return (
        generation.groupby("DATE_TIME")["SOURCE_KEY"]
        .nunique()
        .rename("N_INVERTERS")
        .reset_index()
    )


def load_and_validate_data(
    path: Path = config.MERGED_PATH,
    *,
    write_report: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the merged dataset, validate it and flag partial inverter reporting.

    Returns the cleaned frame and a JSON-serialisable validation report.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}\n"
            "Run `python merge_data.py` or place solar_plant1_merged.csv "
            "in the data folder."
        )

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise DataValidationError(f"Missing required columns: {missing_columns}")

    df["DATE_TIME"] = parse_datetime(df["DATE_TIME"])
    invalid_timestamps = int(df["DATE_TIME"].isna().sum())
    df = df.dropna(subset=["DATE_TIME"])

    for column in MEASUREMENT_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values("DATE_TIME", kind="stable")
    duplicate_timestamps = int(df["DATE_TIME"].duplicated().sum())
    df = df.drop_duplicates("DATE_TIME", keep="first").reset_index(drop=True)

    if df.empty:
        raise DataValidationError("No valid rows remain after timestamp validation.")

    # Derive calendar columns from the timestamp so they can never disagree with it.
    df["HOUR"] = df["DATE_TIME"].dt.hour
    df["DAY"] = df["DATE_TIME"].dt.day
    df["MONTH"] = df["DATE_TIME"].dt.month

    # Inverter reporting completeness.
    counts_source: str | None = None
    if "N_INVERTERS" in df.columns:
        df["N_INVERTERS"] = pd.to_numeric(df["N_INVERTERS"], errors="coerce")
        counts_source = "merged file"
    elif config.RAW_GENERATION_PATH.exists():
        counts = _inverter_counts_from_raw(config.RAW_GENERATION_PATH)
        df = df.merge(counts, on="DATE_TIME", how="left", validate="one_to_one")
        counts_source = "raw generation file"
    else:
        df["N_INVERTERS"] = np.nan

    df["PARTIAL_INVERTER_REPORTING"] = False
    if df["N_INVERTERS"].notna().any():
        n_inverters = df["N_INVERTERS"].fillna(0).astype(int)
        df["N_INVERTERS"] = n_inverters
        df["PARTIAL_INVERTER_REPORTING"] = n_inverters < config.EXPECTED_INVERTERS

    df["PARTIAL_DAYTIME_ROW"] = df["PARTIAL_INVERTER_REPORTING"] & (
        df["IRRADIATION"] > config.PARTIAL_IRRADIATION_THRESHOLD
    )

    low_t, high_t = config.TEMP_LIMITS
    quality = {
        "negative_irradiation": int((df["IRRADIATION"] < 0).sum()),
        "irradiation_above_limit": int((df["IRRADIATION"] > config.MAX_IRRADIATION).sum()),
        "negative_ac_power": int((df["AC_POWER"] < 0).sum()),
        "temperature_out_of_range": int(
            (~df["MODULE_TEMPERATURE"].between(low_t, high_t)
             | ~df["AMBIENT_TEMPERATURE"].between(low_t, high_t)).sum()
        ),
        "night_rows_with_power": int(
            ((df["IRRADIATION"] <= 0.001) & (df["AC_POWER"] > 0)).sum()
        ),
    }

    report: dict[str, Any] = {
        "rows_after_timestamp_validation": int(len(df)),
        "columns": list(df.columns),
        "date_start": str(df["DATE_TIME"].min()),
        "date_end": str(df["DATE_TIME"].max()),
        "invalid_timestamps_removed": invalid_timestamps,
        "duplicate_timestamps_removed": duplicate_timestamps,
        "missing_values": {c: int(v) for c, v in df.isna().sum().items()},
        "quality_checks": quality,
        "expected_inverters": config.EXPECTED_INVERTERS,
        "inverter_count_source": counts_source,
        "timestamps_with_partial_reporting": (
            int(df["PARTIAL_INVERTER_REPORTING"].sum()) if counts_source else None
        ),
        "partial_daytime_rows_flagged": int(df["PARTIAL_DAYTIME_ROW"].sum()),
        "features": FEATURES,
        "target": TARGET,
        "notes": [
            "Original data files are not modified.",
            "Legitimate zero-power readings are retained.",
            "Flagged partial daytime rows and physically invalid rows are "
            "excluded from modelling.",
        ],
    }
    if counts_source is None:
        report["notes"].append(
            "No inverter counts available: partial-reporting rows could not be flagged."
        )

    if write_report:
        config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        config.VALIDATION_REPORT_PATH.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return df, report


def get_modeling_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (X, y, modeling_df) with invalid and partial rows removed."""
    needed = BASE_FEATURES + [TARGET, "DATE_TIME"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns for modelling: {missing}")

    low_t, high_t = config.TEMP_LIMITS
    mask = df[BASE_FEATURES + [TARGET]].notna().all(axis=1)
    mask &= df["IRRADIATION"].between(0.0, config.MAX_IRRADIATION)
    mask &= df["MODULE_TEMPERATURE"].between(low_t, high_t)
    mask &= df["AMBIENT_TEMPERATURE"].between(low_t, high_t)
    mask &= df[TARGET] >= 0
    if "PARTIAL_DAYTIME_ROW" in df.columns:
        mask &= ~df["PARTIAL_DAYTIME_ROW"].astype(bool)

    modeling_df = df.loc[mask].sort_values("DATE_TIME").reset_index(drop=True)
    engineered = engineer_features(modeling_df)
    X = engineered[FEATURES].astype(float)
    y = modeling_df[TARGET].astype(float)
    return X, y, modeling_df


if __name__ == "__main__":
    data, validation = load_and_validate_data(write_report=True)
    X_all, y_all, modeling_data = get_modeling_data(data)

    print("Data validation completed successfully.")
    print(f"Loaded rows: {len(data):,}")
    print(f"Rows used for modelling: {len(modeling_data):,}")
    print(f"Features: {FEATURES}")
    print(f"Date range: {validation['date_start']} to {validation['date_end']}")
    print(f"Partial daytime rows excluded: {validation['partial_daytime_rows_flagged']}")
    print(f"Validation report: {config.VALIDATION_REPORT_PATH}")