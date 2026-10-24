
# Solar Energy Analytics & Power Prediction

A machine learning application for solar energy data analysis and AC power estimation.

## Project Overview

The application analyzes solar generation and weather data and estimates plant-level AC power using an Extra Trees Regressor model.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Extra Trees Regressor
- Streamlit
- Plotly
- Joblib

## Machine Learning

Problem type: Supervised Learning - Regression.

Algorithm: Extra Trees Regressor.

Target variable: AC_POWER.

Input features:
- IRRADIATION
- MODULE_TEMPERATURE
- AMBIENT_TEMPERATURE
- HOUR

The model is evaluated using MAE, RMSE, and R² on a chronological test set.

## Dataset

The project uses the Plant 1 generation and weather sensor CSV files from the Kaggle Solar Power Generation dataset.

Place the original files and the merged dataset inside the data folder.

## Installation

Activate your Python virtual environment, then run:

```powershell
python -m pip install -r requirements.txt
```

## Data Validation

```powershell
python src/data_processing.py
```

## Model Training

```powershell
python src/train_model.py
```

## Run the Application

```powershell
streamlit run app.py
```

## Limitations

The current model estimates AC power at the same timestamp using weather and time features. It does not forecast future power.

The dataset covers a limited period in 2020. Results may not generalize to other seasons or solar plants.

Verify the AC_POWER unit against the original dataset documentation before using the results operationally.
