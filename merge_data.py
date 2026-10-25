"""Merge raw Kaggle generation and weather files into one plant-level dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from reports.src.data_processing import parse_datetime


def _read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    frame["DATE_TIME"] = parse_datetime(frame["DATE_TIME"])
    return frame.dropna(subset=["DATE_TIME"])


def build_merged_dataset(
    generation_path: Path = config.RAW_GENERATION_PATH,
    weather_path: Path = config.RAW_WEATHER_PATH,
    output_path: Path = config.MERGED_PATH,
) -> pd.DataFrame:
    """Sum AC power across inverters, average weather readings, merge on timestamp."""
    generation = _read_csv(generation_path, {"DATE_TIME", "AC_POWER", "SOURCE_KEY"})
    weather_cols = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
    weather = _read_csv(weather_path, {"DATE_TIME", *weather_cols})

    generation["AC_POWER"] = pd.to_numeric(generation["AC_POWER"], errors="coerce")
    for column in weather_cols:
        weather[column] = pd.to_numeric(weather[column], errors="coerce")

    plant_power = (
        generation.groupby("DATE_TIME")
        .agg(AC_POWER=("AC_POWER", "sum"), N_INVERTERS=("SOURCE_KEY", "nunique"))
        .reset_index()
    )
    plant_weather = weather.groupby("DATE_TIME", as_index=False)[weather_cols].mean()

    merged = plant_weather.merge(
        plant_power, on="DATE_TIME", how="inner", validate="one_to_one"
    )
    if merged.empty:
        raise ValueError("The generation and weather files share no timestamps.")

    merged["HOUR"] = merged["DATE_TIME"].dt.hour
    merged["DAY"] = merged["DATE_TIME"].dt.day
    merged["MONTH"] = merged["DATE_TIME"].dt.month
    merged = merged.sort_values("DATE_TIME").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged


def main() -> None:
    merged = build_merged_dataset()
    print("Merged shape:", merged.shape)
    print("\nFirst 5 rows:\n", merged.head())
    print("\nMissing values:\n", merged.isna().sum())
    print("\nTime range:", merged["DATE_TIME"].min(), "to", merged["DATE_TIME"].max())
    print(f"\nSaved: {config.MERGED_PATH}")


if __name__ == "__main__":
    main()