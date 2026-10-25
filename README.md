# Solar Energy Analytics & AC Power Estimation

A Streamlit dashboard and machine-learning pipeline that analyses Plant 1 solar generation and weather data (Kaggle *Solar Power Generation Data*) and estimates plant-level AC power with an Extra Trees Regressor.

> The model estimates power **at the same timestamp** from weather and time inputs. It does not forecast future generation.

## Project layout

```
solar-energy-analytics/
├── app.py                  # Streamlit dashboard (4 pages)
├── requirements.txt
├── Dockerfile
├── .streamlit/config.toml  # theme + server settings
├── config.py               # paths, features, validation limits
├── theme.py                # CSS and Plotly theme
├── merge_data.py           # raw files -> merged dataset
├── reports/
│   ├── prediction.py       # cached model loading + safe inference
│   ├── train_model.py      # training, evaluation, artifact export
│   └── src/
│       └── data_processing.py # loading, validation, feature engineering
├── Data/                   # input CSV files and merged dataset
├── models/                 # generated: model + metadata
└── reports/                # generated: validation report, test predictions
```

## Run locally (Windows PowerShell)

```powershell
cd solar-energy-analytics
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

1. Put `Plant_1_Generation_Data.csv` and `Plant_1_Weather_Sensor_Data.csv` in `data/`.
2. Build the merged dataset: `python merge_data.py`
3. Validate the data (optional): `python -m reports.src.data_processing`
4. Train the model: `python -m reports.train_model`
5. Start the app: `streamlit run app.py`

Steps 2 and 4 can also be done from inside the app with the **Build merged dataset** and **Train the model now** buttons.

Run all commands from the project root. If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

## What changed from the previous version

- One consistent package layout and a single `config.py` (the old code mixed `Data`/`data` and several import roots).
- Shared feature engineering (`TEMP_DELTA`, cyclical `HOUR_SIN`/`HOUR_COS`) used by both training and inference.
- Day-boundary chronological split, plus time-series cross-validation on the training period.
- Cached model loading that refreshes when the model file changes; versioned artifacts reject stale models.
- Stronger validation: duplicate timestamps, physical limits, hour checks, and sanity warnings (night-time irradiation, cold modules in daylight).
- The validation report is only written when requested, not on every app rerun.
- New dashboard theme (sun, sky, wind and foliage palette) with matching Plotly template.

## Deploy

**Streamlit Community Cloud**
1. Push the project to GitHub. Commit `data/solar_plant1_merged.csv` (or both raw files) and, optionally, the trained `models/` files. Remove those entries from `.gitignore` first.
2. In Streamlit Community Cloud choose *New app*, select the repository, branch and `app.py`.
3. If no model is committed, open the app and use **Train the model now** (the filesystem is ephemeral, so commit the model for persistence).

**Docker**
```powershell
docker build -t solar-analytics .
docker run -p 8501:8501 solar-analytics
```
To keep data out of the image, mount it: `-v ${PWD}/data:/app/data -v ${PWD}/models:/app/models`.

For reproducible deployments, pin the versions from `pip freeze` in `requirements.txt` (a model saved with one scikit-learn version should be loaded with the same version).

## Limitations

- Same-timestamp estimation only; no forecasting.
- Data covers a limited period (May to June 2020) at one plant. Results may not generalise to other seasons or plants.
- Confirm the `AC_POWER` unit against the original dataset documentation before operational use.