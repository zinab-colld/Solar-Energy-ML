
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from reports.src.data_processing import (
    FEATURES,
    TARGET,
    load_and_validate_data,
    get_modeling_data,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


def calculate_metrics(y_true, y_pred):
    """Calculate regression evaluation metrics."""

    return {
        "MAE": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "RMSE": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "R2": float(
            r2_score(y_true, y_pred)
        ),
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df, validation_report = load_and_validate_data()
    X, y, modeling_df = get_modeling_data(df)

    if len(modeling_df) < 20:
        raise ValueError(
            "Not enough valid observations to train the model."
        )

    # Chronological 80/20 split. Do not shuffle time-dependent data.
    split_index = int(len(modeling_df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # Baseline 1: predict the training-set mean for every test row.
    mean_predictions = np.full(
        len(y_test),
        y_train.mean()
    )

    mean_metrics = calculate_metrics(
        y_test,
        mean_predictions
    )

    # Baseline 2: linear regression using irradiation only.
    linear_model = LinearRegression()

    linear_model.fit(
        X_train[["IRRADIATION"]],
        y_train
    )

    linear_predictions = linear_model.predict(
        X_test[["IRRADIATION"]]
    )

    linear_metrics = calculate_metrics(
        y_test,
        linear_predictions
    )

    # Primary model: Extra Trees Regressor.
    model = ExtraTreesRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
        max_features=1.0
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    extra_trees_metrics = calculate_metrics(
        y_test,
        predictions
    )

    # Save the trained model and the feature metadata together.
    artifact = {
        "model": model,
        "features": FEATURES,
        "target": TARGET,
        "training_feature_ranges": {
            column: {
                "min": float(X_train[column].min()),
                "max": float(X_train[column].max()),
            }
            for column in FEATURES
        },
        "model_name": "ExtraTreesRegressor",
    }

    model_path = MODEL_DIR / "extra_trees_model.joblib"

    joblib.dump(artifact, model_path)

    # Save held-out predictions for the UI.
    prediction_table = pd.DataFrame({
        "DATE_TIME": (
            modeling_df["DATE_TIME"]
            .iloc[split_index:]
            .astype(str)
            .to_numpy()
        ),
        "ACTUAL_AC_POWER": y_test.to_numpy(),
        "PREDICTED_AC_POWER": predictions,
        "IRRADIATION": (
            X_test["IRRADIATION"].to_numpy()
        ),
    })

    prediction_table.to_csv(
        REPORT_DIR / "test_predictions.csv",
        index=False
    )

    metadata = {
        "model_name": "ExtraTreesRegressor",
        "features": FEATURES,
        "target": TARGET,
        "split_method": "Chronological 80/20; no shuffling",
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "train_start": str(
            modeling_df["DATE_TIME"].iloc[0]
        ),
        "train_end": str(
            modeling_df["DATE_TIME"].iloc[split_index - 1]
        ),
        "test_start": str(
            modeling_df["DATE_TIME"].iloc[split_index]
        ),
        "test_end": str(
            modeling_df["DATE_TIME"].iloc[-1]
        ),
        "partial_daytime_rows_excluded": (
            validation_report["partial_daytime_rows_flagged"]
        ),
        "metrics": {
            "extra_trees": extra_trees_metrics,
            "training_mean_baseline": mean_metrics,
            "irradiation_only_linear_regression": linear_metrics,
        },
        "hyperparameters": model.get_params(),
        "limitations": [
            "This is same-timestamp estimation, not future forecasting.",
            "The dataset covers a limited period in May and June 2020.",
            "Performance may differ on other plants, seasons, and sensors.",
        ],
    }

    metadata_path = MODEL_DIR / "model_metadata.json"

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )

    print("\nTraining completed.")
    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows: {len(X_test):,}")

    for model_name, metrics in metadata["metrics"].items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

    print(f"\nSaved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")
    print(
        f"Saved test predictions: "
        f"{REPORT_DIR / 'test_predictions.csv'}"
    )


if __name__ == "__main__":
    main()
