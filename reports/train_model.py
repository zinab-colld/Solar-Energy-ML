"""Train, evaluate and save the AC power estimation model."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

if __package__ in (None, ""):  # allow `python reports/train_model.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from reports.src.data_processing import (
    BASE_FEATURES,
    FEATURES,
    TARGET,
    get_modeling_data,
    load_and_validate_data,
)

MODEL_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "min_samples_leaf": 2,
    "max_features": 1.0,
    "random_state": config.RANDOM_STATE,
    "n_jobs": -1,
}


def calculate_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """MAE, RMSE and R² for a set of predictions."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train() -> dict[str, Any]:
    """Run the full training pipeline and return the saved metadata."""
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df, validation_report = load_and_validate_data(write_report=True)
    X, y, modeling_df = get_modeling_data(df)

    # Chronological split on a whole-day boundary so no day is shared between sets.
    days = modeling_df["DATE_TIME"].dt.normalize()
    unique_days = np.sort(days.unique())
    if len(unique_days) < config.MIN_TRAINING_DAYS:
        raise ValueError(
            f"Need at least {config.MIN_TRAINING_DAYS} distinct days to train; "
            f"found {len(unique_days)}."
        )
    cutoff = unique_days[int(len(unique_days) * (1 - config.TEST_FRACTION))]
    train_mask = (days < cutoff).to_numpy()

    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
    test_df = modeling_df.loc[~train_mask]
    if X_train.empty or X_test.empty:
        raise ValueError("The chronological split produced an empty train or test set.")

    baselines = {
        "training_mean_baseline": calculate_metrics(
            y_test, np.full(len(y_test), y_train.mean())
        ),
    }
    linear = LinearRegression().fit(X_train[["IRRADIATION"]], y_train)
    baselines["irradiation_only_linear_regression"] = calculate_metrics(
        y_test, linear.predict(X_test[["IRRADIATION"]])
    )

    model = ExtraTreesRegressor(**MODEL_PARAMS).fit(X_train, y_train)
    predictions = np.clip(model.predict(X_test), 0.0, None)
    extra_trees_metrics = calculate_metrics(y_test, predictions)

    # Time-ordered cross-validation on the training portion only.
    cv_r2: list[float] = []
    for fit_idx, val_idx in TimeSeriesSplit(n_splits=5).split(X_train):
        fold = ExtraTreesRegressor(**{**MODEL_PARAMS, "n_estimators": 100})
        fold.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
        cv_r2.append(float(r2_score(y_train.iloc[val_idx], fold.predict(X_train.iloc[val_idx]))))

    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    artifact = {
        "model": model,
        "features": FEATURES,
        "base_features": BASE_FEATURES,
        "target": TARGET,
        "feature_version": config.FEATURE_VERSION,
        "training_feature_ranges": {
            c: {"min": float(X_train[c].min()), "max": float(X_train[c].max())}
            for c in BASE_FEATURES
        },
        "target_range": {"min": float(y_train.min()), "max": float(y_train.max())},
        "model_name": "ExtraTreesRegressor",
        "sklearn_version": sklearn.__version__,
        "trained_at": trained_at,
    }
    joblib.dump(artifact, config.MODEL_PATH, compress=3)

    pd.DataFrame(
        {
            "DATE_TIME": test_df["DATE_TIME"].astype(str).to_numpy(),
            "ACTUAL_AC_POWER": y_test.to_numpy(),
            "PREDICTED_AC_POWER": predictions,
            "IRRADIATION": X_test["IRRADIATION"].to_numpy(),
            "HOUR": X_test["HOUR"].to_numpy(),
        }
    ).to_csv(config.PREDICTIONS_PATH, index=False)

    metadata: dict[str, Any] = {
        "model_name": "ExtraTreesRegressor",
        "trained_at": trained_at,
        "sklearn_version": sklearn.__version__,
        "feature_version": config.FEATURE_VERSION,
        "features": FEATURES,
        "input_features": BASE_FEATURES,
        "target": TARGET,
        "split_method": "Chronological, whole-day boundary, 80/20; no shuffling",
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "train_start": str(modeling_df.loc[train_mask, "DATE_TIME"].iloc[0]),
        "train_end": str(modeling_df.loc[train_mask, "DATE_TIME"].iloc[-1]),
        "test_start": str(test_df["DATE_TIME"].iloc[0]),
        "test_end": str(test_df["DATE_TIME"].iloc[-1]),
        "partial_daytime_rows_excluded": validation_report["partial_daytime_rows_flagged"],
        "metrics": {"extra_trees": extra_trees_metrics, **baselines},
        "cross_validation": {
            "method": "TimeSeriesSplit(5) on training data",
            "r2_folds": cv_r2,
            "r2_mean": float(np.mean(cv_r2)),
            "r2_std": float(np.std(cv_r2)),
        },
        "feature_importances": dict(
            zip(FEATURES, (float(v) for v in model.feature_importances_))
        ),
        "hyperparameters": model.get_params(),
        "limitations": [
            "This is same-timestamp estimation, not future forecasting.",
            "The dataset covers a limited period in May and June 2020.",
            "Performance may differ on other plants, seasons, and sensors.",
        ],
    }
    config.METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    metadata = train()
    print("\nTraining completed.")
    print(f"Training rows: {metadata['training_rows']:,}")
    print(f"Testing rows:  {metadata['testing_rows']:,}")
    for name, metrics in metadata["metrics"].items():
        print(f"\n{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    cv = metadata["cross_validation"]
    print(f"\nTime-series CV R2: {cv['r2_mean']:.4f} ± {cv['r2_std']:.4f}")
    print(f"\nSaved model:       {config.MODEL_PATH}")
    print(f"Saved metadata:    {config.METADATA_PATH}")
    print(f"Saved predictions: {config.PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()