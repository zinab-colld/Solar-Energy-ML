
from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "Data"
REPORT_DIR = ROOT / "reports"

MERGED_PATH = DATA_DIR / "solar_plant1_merged.csv"
GENERATION_PATH = DATA_DIR / "Plant_1_Generation_Data.csv"

FEATURES = [
    "IRRADIATION",
    "MODULE_TEMPERATURE",
    "AMBIENT_TEMPERATURE",
    "HOUR",
]
TARGET = "AC_POWER"
EXPECTED_INVERTERS = 22


def load_and_validate_data():
    """Load the merged data, validate it, and flag partial inverter reporting."""

    if not MERGED_PATH.exists():
        raise FileNotFoundError(
            f"File not found: {MERGED_PATH}\n"
            "Place solar_plant1_merged.csv inside the data folder."
        )

    df = pd.read_csv(MERGED_PATH)
    df.columns = df.columns.str.strip()

    required_columns = [
        "DATE_TIME",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "IRRADIATION",
        "AC_POWER",
        "HOUR",
        "DAY",
        "MONTH",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Parse the timestamp and remove rows with invalid timestamps.
    df["DATE_TIME"] = pd.to_datetime(
        df["DATE_TIME"],
        errors="coerce"
    )

    invalid_timestamps = int(df["DATE_TIME"].isna().sum())

    df = df.dropna(subset=["DATE_TIME"]).copy()
    df = df.sort_values("DATE_TIME").reset_index(drop=True)

    numeric_columns = [
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "IRRADIATION",
        "AC_POWER",
        "HOUR",
        "DAY",
        "MONTH",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    duplicate_timestamps = int(
        df["DATE_TIME"].duplicated().sum()
    )

    # Defaults when the original generation file is unavailable.
    df["N_INVERTERS"] = pd.NA
    df["PARTIAL_INVERTER_REPORTING"] = False
    df["PARTIAL_DAYTIME_ROW"] = False

    partial_timestamp_count = None
    partial_daytime_count = 0

    # Use the raw generation file to count reported inverters.
    if GENERATION_PATH.exists():
        generation = pd.read_csv(GENERATION_PATH)
        generation.columns = generation.columns.str.strip()

        required_raw_columns = {"DATE_TIME", "SOURCE_KEY"}

        if not required_raw_columns.issubset(generation.columns):
            raise ValueError(
                "The raw generation file must contain "
                "DATE_TIME and SOURCE_KEY."
            )

        generation["DATE_TIME"] = pd.to_datetime(
            generation["DATE_TIME"],
            format="%d-%m-%Y %H:%M",
            errors="coerce"
        )

        generation = generation.dropna(
            subset=["DATE_TIME"]
        )

        inverter_counts = (
            generation
            .groupby("DATE_TIME")["SOURCE_KEY"]
            .nunique()
            .rename("N_INVERTERS")
            .reset_index()
        )

        df = df.drop(
            columns=["N_INVERTERS"],
            errors="ignore"
        )

        df = df.merge(
            inverter_counts,
            on="DATE_TIME",
            how="left",
            validate="one_to_one"
        )

        df["N_INVERTERS"] = (
            df["N_INVERTERS"]
            .fillna(0)
            .astype(int)
        )

        df["PARTIAL_INVERTER_REPORTING"] = (
            df["N_INVERTERS"] < EXPECTED_INVERTERS
        )

        # Flag incomplete inverter reporting during meaningful irradiation.
        df["PARTIAL_DAYTIME_ROW"] = (
            df["PARTIAL_INVERTER_REPORTING"]
            & (df["IRRADIATION"] > 0.3)
        )

        partial_timestamp_count = int(
            df["PARTIAL_INVERTER_REPORTING"].sum()
        )

        partial_daytime_count = int(
            df["PARTIAL_DAYTIME_ROW"].sum()
        )

    missing_values = {
        column: int(value)
        for column, value in df.isna().sum().items()
    }

    report = {
        "rows_after_timestamp_validation": int(len(df)),
        "columns": list(df.columns),
        "date_start": str(df["DATE_TIME"].min()),
        "date_end": str(df["DATE_TIME"].max()),
        "invalid_timestamps_removed": invalid_timestamps,
        "duplicate_timestamps": duplicate_timestamps,
        "missing_values": missing_values,
        "expected_inverters": EXPECTED_INVERTERS,
        "timestamps_with_partial_reporting": partial_timestamp_count,
        "partial_daytime_rows_flagged": partial_daytime_count,
        "features": FEATURES,
        "target": TARGET,
        "notes": [
            "Original data files are not modified.",
            "Legitimate zero-power readings are retained.",
            "Flagged partial daytime rows are excluded from modeling."
        ],
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORT_DIR / "data_validation.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    return df, report


def get_modeling_data(df):
    """Prepare features and target for chronological model training."""

    modeling_df = df.dropna(
        subset=FEATURES + [TARGET]
    ).copy()

    # Exclude incomplete daytime plant-power readings.
    if "PARTIAL_DAYTIME_ROW" in modeling_df.columns:
        modeling_df = modeling_df.loc[
            ~modeling_df["PARTIAL_DAYTIME_ROW"].astype(bool)
        ].copy()

    modeling_df = (
        modeling_df
        .sort_values("DATE_TIME")
        .reset_index(drop=True)
    )

    X = modeling_df[FEATURES].astype(float)
    y = modeling_df[TARGET].astype(float)

    return X, y, modeling_df


if __name__ == "__main__":
    data, report = load_and_validate_data()
    X, y, modeling_data = get_modeling_data(data)

    print("Data validation completed successfully.")
    print(f"Loaded rows: {len(data):,}")
    print(f"Rows used for modeling: {len(modeling_data):,}")
    print(f"Features: {FEATURES}")
    print(f"Target: {TARGET}")
    print(f"Date range: {report['date_start']} to {report['date_end']}")
    print(
        "Partial daytime rows excluded:",
        report["partial_daytime_rows_flagged"]
    )
    print(f"Validation report: {REPORT_DIR / 'data_validation.json'}")
