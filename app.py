
from pathlib import Path
import json
import sys

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from reports.src.data_processing import (
    FEATURES,
    load_and_validate_data,
    get_modeling_data,
)
from reports.prediction import predict_ac_power


DATA_PATH = ROOT / "data" / "solar_plant1_merged.csv"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
PREDICTIONS_PATH = ROOT / "reports" / "test_predictions.csv"


st.set_page_config(
    page_title="Solar Energy Analytics",
    page_icon="☀️",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .hero {
        padding: 1.5rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #102a43, #176b66);
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 {
        color: white;
    }
    .hero p {
        color: #e5f4f1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero">
        <h1>☀️ Solar Energy Analytics</h1>
        <p>
            Solar generation insights and machine-learning
            power estimation
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    return load_and_validate_data()


if not DATA_PATH.exists():
    st.error(
        "Dataset not found. Place "
        "solar_plant1_merged.csv inside the data folder."
    )
    st.stop()


try:
    df, validation_report = load_data()
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()


page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Power Estimation",
        "Model Performance",
        "Data Explorer",
    ],
)

st.sidebar.caption(
    "Machine Learning: Extra Trees Regressor"
)

st.sidebar.caption(
    "Task: Same-timestamp power estimation"
)


# --------------------------------------------------
# PAGE 1: OVERVIEW
# --------------------------------------------------

if page == "Overview":

    X, y, modeling_df = get_modeling_data(df)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Merged observations",
        f"{len(df):,}"
    )

    col2.metric(
        "Modeling observations",
        f"{len(modeling_df):,}"
    )

    col3.metric(
        "Input features",
        str(len(FEATURES))
    )

    col4.metric(
        "Flagged daytime rows",
        str(
            validation_report[
                "partial_daytime_rows_flagged"
            ]
        )
    )

    st.subheader("Dataset coverage")

    st.write(
        f"**Start:** {df['DATE_TIME'].min()}  \n"
        f"**End:** {df['DATE_TIME'].max()}"
    )

    fig = px.line(
        df,
        x="DATE_TIME",
        y="AC_POWER",
        title="AC Power by Timestamp",
    )

    fig.update_layout(
        xaxis_title="Timestamp",
        yaxis_title="AC Power (dataset units)",
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(
            df,
            x="DATE_TIME",
            y="IRRADIATION",
            title="Solar Irradiation Over Time",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:
        fig = px.scatter(
            df,
            x="IRRADIATION",
            y="AC_POWER",
            title="Irradiation vs AC Power",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.info(
        "This application estimates power at the same "
        "timestamp. It does not forecast future power."
    )


# --------------------------------------------------
# PAGE 2: POWER ESTIMATION
# --------------------------------------------------

elif page == "Power Estimation":

    st.subheader("Estimate Solar AC Power")

    st.write(
        "Enter weather measurements and the hour of day. "
        "Values outside the training range may produce "
        "less reliable estimates."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            irradiation = st.number_input(
                "Irradiation",
                min_value=0.0,
                max_value=2.0,
                value=0.5,
                step=0.01,
            )

            ambient = st.number_input(
                "Ambient Temperature (°C)",
                min_value=-50.0,
                max_value=100.0,
                value=25.0,
                step=0.5,
            )

        with col2:
            module = st.number_input(
                "Module Temperature (°C)",
                min_value=-50.0,
                max_value=100.0,
                value=30.0,
                step=0.5,
            )

            hour = st.slider(
                "Hour of Day",
                min_value=0,
                max_value=23,
                value=12,
            )

        submitted = st.form_submit_button(
            "Estimate AC Power"
        )

    if submitted:
        try:
            result = predict_ac_power({
                "IRRADIATION": irradiation,
                "MODULE_TEMPERATURE": module,
                "AMBIENT_TEMPERATURE": ambient,
                "HOUR": hour,
            })

            st.metric(
                "Estimated AC Power",
                f"{result['predicted_ac_power']:,.2f}"
            )

            st.caption(
                "Result is displayed in the dataset's "
                "AC_POWER units. Confirm the source unit "
                "before external use."
            )

            for warning in result["warnings"]:
                st.warning(warning)

            if not result["warnings"]:
                st.success(
                    "All inputs are within their "
                    "observed training ranges."
                )

        except FileNotFoundError as error:
            st.error(str(error))

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# PAGE 3: MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    if not METADATA_PATH.exists():

        st.warning(
            "Model metrics are not available yet. "
            "Run: python src/train_model.py"
        )

    else:
        metadata = json.loads(
            METADATA_PATH.read_text(
                encoding="utf-8"
            )
        )

        metrics = metadata.get("metrics", {})

        st.subheader("Test-Set Evaluation")

        st.caption(
            "Results are calculated on the chronological "
            "held-out test set."
        )

        if "extra_trees" in metrics:

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "MAE",
                f"{metrics['extra_trees']['MAE']:.3f}"
            )

            col2.metric(
                "RMSE",
                f"{metrics['extra_trees']['RMSE']:.3f}"
            )

            col3.metric(
                "R²",
                f"{metrics['extra_trees']['R2']:.4f}"
            )

        st.subheader("Model Comparison")

        rows = []

        for name, values in metrics.items():
            rows.append({
                "Model": name.replace("_", " ").title(),
                **values,
            })

        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

        if PREDICTIONS_PATH.exists():

            predictions = pd.read_csv(
                PREDICTIONS_PATH
            )

            fig = px.scatter(
                predictions,
                x="ACTUAL_AC_POWER",
                y="PREDICTED_AC_POWER",
                hover_data=["DATE_TIME", "IRRADIATION"],
                title="Actual vs Predicted AC Power",
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            predictions["RESIDUAL"] = (
                predictions["ACTUAL_AC_POWER"]
                - predictions["PREDICTED_AC_POWER"]
            )

            fig = px.histogram(
                predictions,
                x="RESIDUAL",
                nbins=40,
                title="Residual Error Distribution",
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.info(
                "Test predictions are not available. "
                "Train the model first."
            )

        st.subheader("Model Limitations")

        for limitation in metadata.get(
            "limitations", []
        ):
            st.write(f"- {limitation}")


# --------------------------------------------------
# PAGE 4: DATA EXPLORER
# --------------------------------------------------

elif page == "Data Explorer":

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(100),
        use_container_width=True
    )

    st.subheader("Descriptive Statistics")

    numeric_data = df.select_dtypes(
        include="number"
    )

    st.dataframe(
        numeric_data.describe().T,
        use_container_width=True
    )

    st.subheader("Missing Values")

    missing_values = (
        df.isna()
        .sum()
        .rename("Missing Values")
        .reset_index()
    )

    missing_values.columns = [
        "Column",
        "Missing Values",
    ]

    st.dataframe(
        missing_values,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Column Distribution")

    numeric_columns = numeric_data.columns.tolist()

    selected_column = st.selectbox(
        "Select a numeric column",
        numeric_columns,
    )

    fig = px.histogram(
        df,
        x=selected_column,
        nbins=40,
        title=f"Distribution of {selected_column}",
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
