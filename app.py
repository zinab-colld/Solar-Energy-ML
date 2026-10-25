"""Solar Energy Analytics: Streamlit dashboard for plant data and AC power estimation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import theme
from reports.src.data_processing import FEATURES, get_modeling_data, load_and_validate_data
from merge_data import build_merged_dataset
from reports.prediction import (
    InputValidationError,
    ModelNotAvailableError,
    load_artifact,
    predict_ac_power,
    predict_batch,
)
from reports.train_model import train

P = theme.PALETTE

st.set_page_config(
    page_title="Solar Energy Analytics",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)
theme.inject_css()


# ----------------------------------------------------------------------------
# Cached loaders (keyed on file modification time so retraining refreshes them)
# ----------------------------------------------------------------------------
def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner="Loading and validating dataset…")
def load_dataset(data_mtime: float) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    df, report = load_and_validate_data()
    _, _, modeling_df = get_modeling_data(df)
    return df, report, modeling_df


@st.cache_data
def load_metadata(meta_mtime: float) -> dict[str, Any] | None:
    if not config.METADATA_PATH.exists():
        return None
    return json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_test_predictions(pred_mtime: float) -> pd.DataFrame | None:
    if not config.PREDICTIONS_PATH.exists():
        return None
    frame = pd.read_csv(config.PREDICTIONS_PATH)
    frame["DATE_TIME"] = pd.to_datetime(frame["DATE_TIME"], errors="coerce")
    frame["RESIDUAL"] = frame["ACTUAL_AC_POWER"] - frame["PREDICTED_AC_POWER"]
    return frame


def pretty(name: str) -> str:
    return name.replace("_", " ").title()


# ----------------------------------------------------------------------------
# Shared UI helpers
# ----------------------------------------------------------------------------
def train_button(label: str = "Train the model now", key: str = "train") -> None:
    """Train from inside the app and refresh every cache."""
    if st.button(label, key=key):
        try:
            with st.spinner("Training the model (this takes a few seconds)…"):
                train()
            st.cache_data.clear()
            st.rerun()
        except Exception as error:  # surfaced to the user, never silent
            st.error(f"Training failed: {error}")


def date_filter(df: pd.DataFrame, key: str) -> pd.DataFrame:
    low, high = df["DATE_TIME"].min().date(), df["DATE_TIME"].max().date()
    chosen = st.date_input(
        "Date range", value=(low, high), min_value=low, max_value=high, key=key
    )
    if isinstance(chosen, (tuple, list)):
        if not chosen:
            return df
        start, end = chosen[0], chosen[-1]
    else:
        start = end = chosen
    dates = df["DATE_TIME"].dt.date
    return df.loc[(dates >= start) & (dates <= end)]


def require_model() -> dict[str, Any] | None:
    try:
        return load_artifact()
    except ModelNotAvailableError as error:
        st.warning(str(error))
        train_button(key="train_inline")
    except Exception as error:
        st.error(f"The model could not be loaded: {error}")
        train_button("Retrain the model", key="train_retry")
    return None


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def page_overview(df: pd.DataFrame, report: dict[str, Any], modeling_df: pd.DataFrame) -> None:
    theme.hero(
        "Solar Energy Analytics",
        "Plant-level generation and weather insight, with machine-learning "
        "estimates of AC power at the same timestamp.",
        ["Extra Trees Regressor", "Chronological validation", "Same-timestamp estimation"],
    )
    theme.section_header("Plant overview", "Filter the period to update every chart below.")
    view = date_filter(df, "overview_dates")
    if view.empty:
        st.info("No observations in the selected range.")
        return

    daytime = view[view["IRRADIATION"] > 0.05]
    peak_row = view.loc[view["AC_POWER"].idxmax()]
    theme.kpi_row([
        ("Observations", f"{len(view):,}", f"{len(df):,} in full dataset", "sky"),
        ("Peak AC power", f"{view['AC_POWER'].max():,.0f}", f"on {peak_row['DATE_TIME']:%d %b, %H:%M}", "gold"),
        ("Mean daytime AC power", f"{daytime['AC_POWER'].mean():,.0f}" if len(daytime) else "n/a",
         "irradiation above 0.05", "amber"),
        ("Peak irradiation", f"{view['IRRADIATION'].max():.2f}", "sensor units", "sage"),
        ("Rows excluded", f"{report['partial_daytime_rows_flagged']:,}",
         "partial daytime reporting", "slate_soft"),
    ])

    theme.section_header("Power and irradiation over time")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=view["DATE_TIME"], y=view["AC_POWER"], name="AC power", mode="lines",
        line=dict(color=P["gold"], width=1.6), fill="tozeroy", fillcolor="rgba(245,166,35,0.22)",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=view["DATE_TIME"], y=view["IRRADIATION"], name="Irradiation", mode="lines",
        line=dict(color=P["sky"], width=1.4),
    ), secondary_y=True)
    flagged = view[view["PARTIAL_DAYTIME_ROW"]]
    if not flagged.empty:
        fig.add_trace(go.Scatter(
            x=flagged["DATE_TIME"], y=flagged["AC_POWER"], name="Excluded (partial reporting)",
            mode="markers", marker=dict(color=P["amber"], symbol="x", size=7),
        ), secondary_y=False)
    fig.update_yaxes(title_text="AC power (dataset units)", secondary_y=False)
    fig.update_yaxes(title_text="Irradiation", secondary_y=True, showgrid=False)
    fig.update_layout(hovermode="x unified")
    theme.plot(fig, 400)

    left, right = st.columns(2)
    with left:
        profile = view.groupby("HOUR", as_index=False)[["AC_POWER", "IRRADIATION"]].mean()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=profile["HOUR"], y=profile["AC_POWER"], name="Mean AC power",
                             marker_color=P["gold"]), secondary_y=False)
        fig.add_trace(go.Scatter(x=profile["HOUR"], y=profile["IRRADIATION"], name="Mean irradiation",
                                 mode="lines+markers", line=dict(color=P["sky"], width=2.5)), secondary_y=True)
        fig.update_layout(title="Average daily profile by hour")
        fig.update_xaxes(title_text="Hour of day", dtick=2)
        fig.update_yaxes(title_text="AC power", secondary_y=False)
        fig.update_yaxes(title_text="Irradiation", secondary_y=True, showgrid=False)
        theme.plot(fig, 380)
    with right:
        fig = px.scatter(
            view, x="IRRADIATION", y="AC_POWER", color="MODULE_TEMPERATURE",
            color_continuous_scale=theme.SUN_SCALE, opacity=0.75,
            labels={"MODULE_TEMPERATURE": "Module °C"},
            title="Irradiation vs AC power",
        )
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(coloraxis_colorbar=dict(title="Module °C", thickness=12))
        theme.plot(fig, 380)

    daily = (
        view.assign(DATE=view["DATE_TIME"].dt.date)
        .groupby("DATE", as_index=False)["AC_POWER"].mean()
    )
    fig = px.bar(daily, x="DATE", y="AC_POWER", title="Daily mean AC power",
                 color_discrete_sequence=[P["sage"]])
    fig.update_layout(xaxis_title=None, yaxis_title="AC power")
    theme.plot(fig, 320)

    st.info(
        "This application estimates power at the same timestamp from weather and "
        "time inputs. It does not forecast future generation."
    )


PRESETS: dict[str, dict[str, float]] = {
    "Clear noon": {"in_irr": 0.95, "in_amb": 33.0, "in_mod": 52.0, "in_hour": 12},
    "Hazy morning": {"in_irr": 0.30, "in_amb": 26.0, "in_mod": 30.0, "in_hour": 8},
    "Late afternoon": {"in_irr": 0.25, "in_amb": 32.0, "in_mod": 36.0, "in_hour": 17},
    "Night": {"in_irr": 0.0, "in_amb": 23.0, "in_mod": 22.0, "in_hour": 2},
}
for _key, _value in {"in_irr": 0.5, "in_amb": 25.0, "in_mod": 30.0, "in_hour": 12}.items():
    st.session_state.setdefault(_key, _value)


def _apply_preset(name: str) -> None:
    st.session_state.update(PRESETS[name])


def page_estimation() -> None:
    theme.hero(
        "Power Estimation",
        "Enter weather readings and the hour of day to estimate plant AC power.",
        ["Validated inputs", "Training-range checks"],
    )
    artifact = require_model()
    if artifact is None:
        return
    ranges = artifact["training_feature_ranges"]

    theme.section_header("Quick scenarios", "Load typical conditions, then adjust the values.")
    for column, name in zip(st.columns(len(PRESETS)), PRESETS):
        column.button(name, key=f"preset_{name}", on_click=_apply_preset, args=(name,))

    def span(feature: str) -> str:
        return f"Training range: {ranges[feature]['min']:.2f} to {ranges[feature]['max']:.2f}"

    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            irradiation = st.number_input(
                "Irradiation", 0.0, config.MAX_IRRADIATION, step=0.01, key="in_irr",
                help=span("IRRADIATION"),
            )
            ambient = st.number_input(
                "Ambient temperature (°C)", config.TEMP_LIMITS[0], config.TEMP_LIMITS[1],
                step=0.5, key="in_amb", help=span("AMBIENT_TEMPERATURE"),
            )
        with right:
            module = st.number_input(
                "Module temperature (°C)", config.TEMP_LIMITS[0], config.TEMP_LIMITS[1],
                step=0.5, key="in_mod", help=span("MODULE_TEMPERATURE"),
            )
            hour = st.slider("Hour of day", 0, 23, key="in_hour")
        submitted = st.form_submit_button("Estimate AC power")

    if submitted:
        try:
            result = predict_ac_power({
                "IRRADIATION": irradiation, "MODULE_TEMPERATURE": module,
                "AMBIENT_TEMPERATURE": ambient, "HOUR": hour,
            })
            st.session_state["last_estimate"] = result
        except InputValidationError as error:
            st.session_state.pop("last_estimate", None)
            st.error(str(error))
        except Exception as error:
            st.session_state.pop("last_estimate", None)
            st.error(f"Prediction failed: {error}")

    result = st.session_state.get("last_estimate")
    if result is None:
        st.info("Choose a scenario or enter values, then select Estimate AC power.")
        return

    target_max = artifact["target_range"]["max"]
    value = result.predicted_ac_power
    left, right = st.columns([1, 1.25])
    with left:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=value,
            number=dict(valueformat=",.1f", font=dict(family=theme.HEADING_STACK, size=40)),
            title=dict(text="Estimated AC power"),
            gauge=dict(
                axis=dict(range=[0, max(target_max, value)], tickcolor=P["muted"]),
                bar=dict(color=P["gold"], thickness=0.3),
                bgcolor="white", borderwidth=0,
                steps=[
                    dict(range=[0, target_max * 0.33], color=P["sky_light"]),
                    dict(range=[target_max * 0.33, target_max * 0.66], color=P["mint"]),
                    dict(range=[target_max * 0.66, max(target_max, value)], color=P["sand"]),
                ],
            ),
        ))
        gauge.update_layout(margin=dict(l=20, r=20, t=70, b=10))
        theme.plot(gauge, 320)
    with right:
        theme.kpi_row([
            ("Share of observed peak", f"{value / target_max:.0%}" if target_max else "n/a",
             "relative to training maximum", "sage"),
            ("Input status", "In range" if result.in_training_range else "Outside range",
             "vs training data", "sky" if result.in_training_range else "amber"),
        ])
        st.caption("Result is in the dataset's AC_POWER units. Confirm the source unit before external use.")
        for message in result.warnings:
            st.warning(message)
        if not result.warnings:
            st.success("All inputs are within their observed training ranges.")

    sweep_max = max(ranges["IRRADIATION"]["max"] * 1.05, result.inputs["IRRADIATION"] * 1.05, 0.2)
    sweep = pd.DataFrame({"IRRADIATION": np.linspace(0, min(sweep_max, config.MAX_IRRADIATION), 60)})
    for feature in ("MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE", "HOUR"):
        sweep[feature] = result.inputs[feature]
    sweep["ESTIMATE"] = predict_batch(sweep, artifact)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep["IRRADIATION"], y=sweep["ESTIMATE"], mode="lines", name="Estimate",
                             line=dict(color=P["sky"], width=3)))
    fig.add_trace(go.Scatter(x=[result.inputs["IRRADIATION"]], y=[value], mode="markers", name="Your input",
                             marker=dict(color=P["amber"], size=14, symbol="star", line=dict(color="white", width=1))))
    fig.update_layout(title="Sensitivity to irradiation (other inputs held constant)",
                      xaxis_title="Irradiation", yaxis_title="Estimated AC power")
    theme.plot(fig, 340)


def page_performance() -> None:
    theme.hero(
        "Model Performance",
        "How well the model estimates AC power on days it never saw during training.",
        ["Held-out test days", "Baseline comparison"],
    )
    metadata = load_metadata(_mtime(config.METADATA_PATH))
    if metadata is None:
        st.warning("Model metrics are not available yet.")
        train_button(key="train_perf")
        return

    metrics = metadata.get("metrics", {})
    et = metrics.get("extra_trees")
    cv = metadata.get("cross_validation", {})
    if et:
        theme.kpi_row([
            ("MAE", f"{et['MAE']:.2f}", "mean absolute error", "gold"),
            ("RMSE", f"{et['RMSE']:.2f}", "penalises large misses", "sky"),
            ("R²", f"{et['R2']:.4f}", "variance explained on test days", "sage"),
            ("Time-series CV R²", f"{cv['r2_mean']:.3f}" if cv else "n/a",
             f"± {cv['r2_std']:.3f} across folds" if cv else "", "amber"),
        ])

    theme.section_header("Model comparison", "Lower MAE and RMSE are better; higher R² is better.")
    rows = [{"Model": pretty(name), **vals} for name, vals in metrics.items()]
    comparison = pd.DataFrame(rows)
    left, right = st.columns([1, 1.2])
    with left:
        theme.table(comparison, hide_index=True)
    with right:
        fig = px.bar(comparison, x="Model", y="RMSE", color="Model", title="Test RMSE by model",
                     color_discrete_sequence=[P["gold"], P["sky"], P["sage"]])
        fig.update_layout(showlegend=False, xaxis_title=None)
        theme.plot(fig, 300)

    predictions = load_test_predictions(_mtime(config.PREDICTIONS_PATH))
    if predictions is None:
        st.info("Test predictions are not available. Train the model first.")
    else:
        left, right = st.columns(2)
        with left:
            top = float(max(predictions["ACTUAL_AC_POWER"].max(), predictions["PREDICTED_AC_POWER"].max()))
            fig = px.scatter(predictions, x="ACTUAL_AC_POWER", y="PREDICTED_AC_POWER",
                             hover_data=["DATE_TIME", "IRRADIATION"], opacity=0.7,
                             title="Actual vs predicted AC power", color_discrete_sequence=[P["sky"]])
            fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="Perfect fit",
                                     line=dict(color=P["amber"], dash="dash")))
            theme.plot(fig, 380)
        with right:
            fig = px.histogram(predictions, x="RESIDUAL", nbins=40, title="Residual error distribution",
                               color_discrete_sequence=[P["sage"]])
            fig.update_layout(xaxis_title="Actual − predicted", yaxis_title="Count")
            theme.plot(fig, 380)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=predictions["DATE_TIME"], y=predictions["ACTUAL_AC_POWER"], name="Actual",
                                 mode="lines", line=dict(color=P["slate_soft"], width=1.5)))
        fig.add_trace(go.Scatter(x=predictions["DATE_TIME"], y=predictions["PREDICTED_AC_POWER"], name="Predicted",
                                 mode="lines", line=dict(color=P["gold"], width=1.5)))
        fig.update_layout(title="Held-out period: actual vs predicted", hovermode="x unified",
                          yaxis_title="AC power")
        theme.plot(fig, 340)

    importances = metadata.get("feature_importances")
    if importances:
        ordered = sorted(importances.items(), key=lambda kv: kv[1])
        fig = go.Figure(go.Bar(
            x=[v for _, v in ordered], y=[pretty(k) for k, _ in ordered], orientation="h",
            marker_color=P["gold"],
        ))
        fig.update_layout(title="Feature importance", xaxis_title="Relative importance")
        theme.plot(fig, 320)

    theme.section_header("Training details")
    theme.kpi_row([
        ("Training rows", f"{metadata['training_rows']:,}", f"{metadata['train_start'][:10]} to {metadata['train_end'][:10]}", "sky"),
        ("Testing rows", f"{metadata['testing_rows']:,}", f"{metadata['test_start'][:10]} to {metadata['test_end'][:10]}", "sage"),
        ("Trained at (UTC)", metadata.get("trained_at", "n/a")[:16].replace("T", " "), f"scikit-learn {metadata.get('sklearn_version', '?')}", "gold"),
    ])
    with st.expander("Hyperparameters"):
        st.json(metadata.get("hyperparameters", {}))
    theme.section_header("Model limitations")
    for item in metadata.get("limitations", []):
        st.write(f"- {item}")
    train_button("Retrain the model", key="train_again")


def page_explorer(df: pd.DataFrame, report: dict[str, Any], modeling_df: pd.DataFrame) -> None:
    theme.hero("Data Explorer", "Inspect, filter and download the merged plant dataset.",
               ["Quality checks", "Distributions", "Correlations"])
    tab_preview, tab_stats, tab_quality, tab_dist = st.tabs(
        ["Preview", "Statistics", "Data quality", "Distributions"]
    )

    with tab_preview:
        view = date_filter(df, "explorer_dates")
        only_model = st.toggle("Show only rows used for modelling", value=False)
        if only_model:
            view = view[view["DATE_TIME"].isin(modeling_df["DATE_TIME"])]
        st.caption(f"{len(view):,} rows")
        theme.table(view.head(500), hide_index=True)
        st.download_button("Download filtered data (CSV)", view.to_csv(index=False).encode("utf-8"),
                           file_name="solar_filtered.csv", mime="text/csv")

    numeric = df.select_dtypes(include="number")
    with tab_stats:
        theme.table(numeric.describe().T)

    with tab_quality:
        theme.kpi_row([
            ("Rows loaded", f"{report['rows_after_timestamp_validation']:,}", "after timestamp checks", "sky"),
            ("Duplicate timestamps", str(report["duplicate_timestamps_removed"]), "removed", "amber"),
            ("Invalid timestamps", str(report["invalid_timestamps_removed"]), "removed", "amber"),
            ("Partial daytime rows", str(report["partial_daytime_rows_flagged"]), "excluded from modelling", "gold"),
        ])
        left, right = st.columns(2)
        with left:
            theme.section_header("Missing values")
            missing = pd.Series(report["missing_values"], name="Missing").rename_axis("Column").reset_index()
            theme.table(missing, hide_index=True)
        with right:
            theme.section_header("Physical checks")
            checks = pd.Series(report["quality_checks"], name="Rows").rename_axis("Check").reset_index()
            checks["Check"] = checks["Check"].map(pretty)
            theme.table(checks, hide_index=True)
        with st.expander("Full validation report"):
            st.json(report)

    with tab_dist:
        column = st.selectbox("Numeric column", numeric.columns.tolist(),
                              index=numeric.columns.tolist().index("AC_POWER") if "AC_POWER" in numeric else 0)
        fig = px.histogram(df, x=column, nbins=40, title=f"Distribution of {pretty(column)}",
                           color_discrete_sequence=[P["gold"]])
        theme.plot(fig, 340)
        corr_cols = [c for c in FEATURES[:3] + ["AC_POWER"] if c in df.columns] + ["HOUR"]
        corr = df[corr_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale=theme.CORR_SCALE, zmin=-1, zmax=1,
                        title="Correlation matrix", aspect="auto")
        theme.plot(fig, 420)


# ----------------------------------------------------------------------------
# Data bootstrap and navigation
# ----------------------------------------------------------------------------
def ensure_dataset() -> None:
    if config.MERGED_PATH.exists():
        return
    theme.hero("Solar Energy Analytics", "The merged dataset is missing.")
    if config.RAW_GENERATION_PATH.exists() and config.RAW_WEATHER_PATH.exists():
        st.info("Raw plant files were found. Build the merged dataset to continue.")
        if st.button("Build merged dataset"):
            try:
                with st.spinner("Merging generation and weather data…"):
                    build_merged_dataset()
                st.cache_data.clear()
                st.rerun()
            except Exception as error:
                st.error(f"Could not merge the raw files: {error}")
    else:
        st.error(
            "Dataset not found. Place `Plant_1_Generation_Data.csv` and "
            "`Plant_1_Weather_Sensor_Data.csv` (or `solar_plant1_merged.csv`) "
            f"inside `{config.DATA_DIR}`, then reload."
        )
    st.stop()


def sidebar(model_ready: bool, metadata: dict[str, Any] | None) -> str:
    st.sidebar.markdown(
        '<div class="brand"><div class="sun"></div><div><div class="name">Solar Analytics</div>'
        '<div class="tag">AC power estimation</div></div></div>',
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Power Estimation", "Model Performance", "Data Explorer"],
        label_visibility="collapsed",
    )
    chip = (
        '<span class="chip ok">Model ready</span>' if model_ready
        else '<span class="chip warn">Model not trained</span>'
    )
    st.sidebar.markdown(chip, unsafe_allow_html=True)
    if metadata:
        st.sidebar.caption(f"Test R²: {metadata['metrics']['extra_trees']['R2']:.3f}")
    st.sidebar.caption("Task: same-timestamp power estimation")
    return page


def main() -> None:
    ensure_dataset()
    try:
        df, report, modeling_df = load_dataset(_mtime(config.MERGED_PATH))
    except Exception as error:
        st.error(f"Could not load the dataset: {error}")
        st.stop()

    metadata = load_metadata(_mtime(config.METADATA_PATH))
    try:
        load_artifact()
        model_ready = True
    except Exception:
        model_ready = False

    page = sidebar(model_ready, metadata)
    if page == "Overview":
        page_overview(df, report, modeling_df)
    elif page == "Power Estimation":
        page_estimation()
    elif page == "Model Performance":
        page_performance()
    else:
        page_explorer(df, report, modeling_df)


main()