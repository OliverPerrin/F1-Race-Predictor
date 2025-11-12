"""Streamlit dashboard for inspecting and simulating F1 qualifying predictions."""

from __future__ import annotations

import os
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt

try:
    import fastf1  # type: ignore
except Exception:  # pragma: no cover - FastF1 may be unavailable on some deployments
    fastf1 = None

# Configure Streamlit up front so the page always opens with the same look and feel.

@st.cache_data(show_spinner=False)
def load_event_schedule(year: int) -> pd.DataFrame:
    """Fetch the FIA event schedule for a given season if FastF1 is available."""
    if fastf1 is None:
        raise RuntimeError("FastF1 is not installed in this environment.")

    schedule = fastf1.get_event_schedule(year)
    event_format = schedule.get("EventFormat")
    if event_format is not None:
        schedule = schedule[event_format != "testing"].copy()
    else:
        schedule = schedule.copy()
    return schedule

st.set_page_config(page_title="F1 Race Predictor", layout="wide")
# Adopt a clean seaborn theme so every chart is readable by default.
sns.set_theme(style="whitegrid")

BASE_DIR = os.path.dirname(__file__)
SAMPLE_DATA_DIR = os.path.join(BASE_DIR, "sample_data")
FALLBACK_DATA_PATH = os.path.join(SAMPLE_DATA_DIR, "processed_data.csv")
FALLBACK_RESULTS_PATH = os.path.join(SAMPLE_DATA_DIR, "results.csv")


# --------- Cached helpers ---------
@st.cache_data(show_spinner=True)
def load_dataset(path: str) -> pd.DataFrame:
    """Read the processed dataset from disk (cached for snappy reloads)."""
    return pd.read_csv(path)


@st.cache_resource(show_spinner=True)
def load_model(path: str):
    """Load a persisted sklearn model if it exists on disk."""
    if os.path.exists(path):
        return joblib.load(path)
    return None

# This is just to get it working with streamlit cloud
def resolve_existing_path(primary: str, fallback: str | None = None, *, required: bool) -> tuple[str | None, bool]:
    """Return the first existing path and flag whether a bundled fallback was used."""
    if primary and os.path.exists(primary):
        return primary, False
    if fallback and os.path.exists(fallback):
        return fallback, True
    if required:
        missing = primary or fallback or ""
        raise FileNotFoundError(f"Required file missing: {missing}")
    return None, False


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Recreate the feature matrix used during training so predictions align."""
    feature_df = df.select_dtypes(include=["number"]).copy()
    feature_df = feature_df.drop(columns=["y_position", "y_q3", "Position"], errors="ignore")
    feature_df = feature_df.dropna(axis=1, how="all")
    return feature_df


def create_upcoming_dataset(
    history_df: pd.DataFrame,
    candidate_years: Iterable[int],
    upcoming_year: int,
    upcoming_race: str,
    lookback: int,
    upcoming_round: int | None = None,
) -> pd.DataFrame:
    """Average the recent form for each driver to project a future qualifying weekend."""
    candidate_years = set(candidate_years)
    average_columns = [
        col
        for col in ["Q1", "Q2", "Q3", "BestLap", "AvgLap", "Improvement"]
        if col in history_df.columns
    ]

    scenario_rows = []
            # Schedule helper to be inserted here in the next patch

    # Ensure we can grab the most recent races per driver after filtering
    sorted_history = history_df.sort_values(["DriverNumber", "Year", "RoundNumber"])
    for driver_number, driver_history in sorted_history.groupby("DriverNumber"):
        trimmed = (
            driver_history[driver_history["Year"].isin(candidate_years)]
            if candidate_years
            else driver_history
        )
        if trimmed.empty:
            trimmed = driver_history
        recent = trimmed.tail(lookback)  # Use the latest `lookback` rounds to estimate future form

        latest_row = recent.iloc[-1].copy()
        for column in average_columns:
            latest_row[column] = recent[column].mean(skipna=True)

        latest_row["Year"] = upcoming_year
        latest_row["RaceName"] = upcoming_race
        if "RoundNumber" in latest_row:
            if upcoming_round is not None:
                latest_row["RoundNumber"] = int(upcoming_round)
            elif recent["RoundNumber"].notna().any():
                latest_row["RoundNumber"] = int(recent["RoundNumber"].dropna().max()) + 1
            else:
                latest_row["RoundNumber"] = 1

        for target_col in ["Position", "y_position", "y_q3"]:
            if target_col in latest_row:
                latest_row[target_col] = pd.NA

        scenario_rows.append(latest_row)

    return pd.DataFrame(scenario_rows)


def derive_driver_labels(df: pd.DataFrame) -> pd.Series:
    """Build readable driver labels even when name fields are missing."""
    last_names = df.get("LastName", pd.Series(index=df.index, dtype="object")).astype("string").fillna("")
    first_names = df.get("FirstName", pd.Series(index=df.index, dtype="object")).astype("string").fillna("")
    driver_numbers = df.get("DriverNumber", pd.Series(index=df.index, dtype="object")).astype("string").fillna("")

    labels = last_names.mask(last_names.str.strip() == "", first_names)
    labels = labels.mask(labels.str.strip() == "", driver_numbers)
    # Generate numeric fallbacks so we never render a blank driver label
    fallback = pd.Series(range(len(df)), index=df.index).astype(str)
    labels = labels.mask(labels.str.strip() == "", fallback)
    return labels.astype(str)


@st.cache_data(show_spinner=True)
def load_prediction_results(path: str | None) -> pd.DataFrame:
    """Load stored model predictions for computing offline diagnostics."""
    if path and os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["target", "actual", "predicted"])


def calculate_regression_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Compute common regression diagnostics for a prediction set."""
    residuals = df["predicted"] - df["actual"]
    abs_errors = residuals.abs()
    squared_errors = residuals.pow(2)
    mae = abs_errors.mean()
    rmse = squared_errors.mean() ** 0.5
    # Guard against divide-by-zero when real finishing positions are missing
    mape = (abs_errors.div(df["actual"].replace(0, pd.NA))).dropna().mean() * 100 if df["actual"].notna().any() else pd.NA
    bias = residuals.mean()
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "Mean Bias": bias,
    }


def calculate_classification_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Calculate accuracy, precision, recall, and F1 for a binary classifier."""
    actual = df["actual"].astype(int)
    predicted = df["predicted"].astype(int)
    # Tally confusion-matrix counts manually so downstream metrics remain transparent
    tp = int(((predicted == 1) & (actual == 1)).sum())
    tn = int(((predicted == 0) & (actual == 0)).sum())
    fp = int(((predicted == 1) & (actual == 0)).sum())
    fn = int(((predicted == 0) & (actual == 1)).sum())

    total = len(df)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "True Positives": tp,
        "False Positives": fp,
        "True Negatives": tn,
        "False Negatives": fn,
    }


# --------- Load data + trained models ---------
DATA_PATH = "data/processed/processed_data.csv"
PREDICTIONS_DIR = "data/predictions"
RESULTS_PATH = "data/predictions/results.csv"

try:
    dataset_path, used_fallback_dataset = resolve_existing_path(
        DATA_PATH, FALLBACK_DATA_PATH, required=True
    )
except FileNotFoundError:
    st.error(
        "No processed dataset is available. Run `python src/preprocessing.py` to regenerate it "
        "or bundle a CSV at `src/sample_data/processed_data.csv` before deploying."
    )
    st.stop()

df = load_dataset(dataset_path)
if used_fallback_dataset:
    st.info(
        "Bundled sample dataset loaded (streamlit cloud fallback). Regenerate `data/processed/processed_data.csv` "
        "for the latest results."
    )

position_model = load_model(os.path.join(PREDICTIONS_DIR, "position_regressor.pkl"))
q3_model = load_model(os.path.join(PREDICTIONS_DIR, "q3_classifier.pkl"))
top10_model = load_model(os.path.join(PREDICTIONS_DIR, "top10_classifier.pkl"))


# --------- Sidebar controls ---------
st.sidebar.header("Filter and simulate weekends")

available_years = sorted(df["Year"].unique())
default_years = available_years[-2:] if len(available_years) >= 2 else available_years
selected_years = st.sidebar.multiselect(
    "Seasons",
    options=available_years,
    default=default_years,
    help="Choose which championship years should contribute to the analysis.",
)

history_df = df[df["Year"].isin(selected_years)] if selected_years else df.copy()

mode = st.sidebar.radio(
    "Data mode",
    options=["Historical weekends", "Upcoming weekend"],
    help="Switch between analysing past sessions and projecting the next race.",
)

if mode == "Historical weekends":
    filtered_df = history_df
    race_options = ["All rounds"] + sorted(filtered_df["RaceName"].unique())
    selected_race = st.sidebar.selectbox(
        "Race weekend",
        options=race_options,
        help="Focus on one event or view every round from the selected seasons.",
    )
    if selected_race != "All rounds":
        filtered_df = filtered_df[filtered_df["RaceName"] == selected_race]

    future_only = st.sidebar.checkbox(
        "Show entries without classified results",
        value=False,
        help="Enable this if you only want to see sessions that have not run yet.",
    )
    if future_only:
        filtered_df = filtered_df[filtered_df["Position"].isna()]
else:
    upcoming_year_default = max(available_years) if available_years else 2025
    upcoming_year = st.sidebar.number_input(
        "Upcoming season",
        min_value=2000,
        max_value=2100,
        value=upcoming_year_default,
        help="Enter the season you want to project (e.g. 2025).",
    )
    schedule_df = pd.DataFrame()
    schedule_warning: str | None = None
    if fastf1 is not None:
        try:
            schedule_df = load_event_schedule(int(upcoming_year))
        except Exception as exc:  # pragma: no cover - defensive against remote API issues
            schedule_warning = str(exc)
    else:
        schedule_warning = "FastF1 package is unavailable; enter upcoming races manually."

    completed_round = (
        history_df[
            (history_df["Year"] == upcoming_year) & history_df["Position"].notna()
        ]["RoundNumber"].max()
    )
    completed_round = int(completed_round) if pd.notna(completed_round) else 0

    race_options: list[str] = []
    race_round_map: dict[str, int] = {}
    if not schedule_df.empty and {"EventName", "RoundNumber"}.issubset(schedule_df.columns):
        remaining = schedule_df[schedule_df["RoundNumber"] > completed_round]
        if remaining.empty:
            remaining = schedule_df
        race_options = remaining["EventName"].astype(str).tolist()
        race_round_map = {
            str(row.EventName): int(row.RoundNumber)
            for row in remaining.itertuples()
            if pd.notna(row.RoundNumber)
        }
    if schedule_warning and not race_options:
        st.sidebar.caption(schedule_warning)

    custom_label = "Custom entry"
    upcoming_race: str
    upcoming_round: int
    if race_options:
        race_selection = st.sidebar.selectbox(
            "Upcoming race",
            options=race_options + [custom_label],
            index=0,
            help="Pick any remaining event this season or choose custom to simulate another scenario.",
        )
        if race_selection == custom_label:
            default_label = race_options[0] if race_options else "Upcoming Grand Prix"
            upcoming_race = st.sidebar.text_input(
                "Race name",
                value=default_label,
                help="Name the race so predictions are clearly labelled.",
            )
            upcoming_round = st.sidebar.number_input(
                "Projected round number",
                min_value=1,
                max_value=50,
                value=max(completed_round + 1, 1),
                help="Adjust if you want to model events further into the future.",
            )
        else:
            upcoming_race = race_selection
            upcoming_round = race_round_map.get(race_selection, max(completed_round + 1, 1))
    else:
        upcoming_race = st.sidebar.text_input(
            "Race name",
            value="Upcoming Grand Prix",
            help="Name the race so predictions are clearly labelled.",
        )
        upcoming_round = st.sidebar.number_input(
            "Projected round number",
            min_value=1,
            max_value=50,
            value=max(completed_round + 1, 1),
            help="Adjust if you want to model events further into the future.",
        )
    lookback = st.sidebar.slider(
        "Number of recent rounds to average",
        min_value=1,
        max_value=10,
        value=3,
        help="The model will average this many recent results per driver to craft the scenario.",
    )
    filtered_df = create_upcoming_dataset(
        history_df,
        selected_years,
        upcoming_year,
        upcoming_race,
        lookback,
        upcoming_round=upcoming_round,
    )

if filtered_df.empty:
    st.warning("No rows match the current selections. Try broadening the filters.")
    st.stop()


# --------- Prediction pipeline ---------
# Build the same numeric feature set the models expect before calling `.predict`
features = build_feature_matrix(filtered_df)

st.title("🏎️ F1 Race Predictor Dashboard")
st.caption(
    "Slice past weekends or assemble an upcoming scenario to see how the models rate every driver."
)


def write_prediction_section(
    title: str,
    predictions: pd.Series,
    extra_columns: pd.DataFrame,
    probability_df: pd.DataFrame | None = None,
) -> None:
    """Render a tidy table and optional probability chart for a given model."""

    st.subheader(title)
    table_df = pd.concat([extra_columns.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)
    st.dataframe(table_df, width="stretch", hide_index=True)

    if probability_df is not None and not probability_df.empty:
        st.markdown("**Model confidence**")
        # Convert to pure floats so the stacked bars render without dtype surprises
        probability_numeric = probability_df.astype(float).fillna(0.0)
        driver_labels = probability_numeric.index.astype(str).tolist()
        x_positions = np.arange(len(driver_labels))
        stack_base = np.zeros(len(driver_labels))
        fig, ax = plt.subplots(figsize=(max(8, len(driver_labels) * 0.45), 4))
        colors = sns.color_palette("coolwarm", n_colors=len(probability_numeric.columns))
        for column, color in zip(probability_numeric.columns, colors):
            # Stack each class' probability so bars always sum to one per driver
            values = probability_numeric[column].to_numpy()
            ax.bar(x_positions, values, bottom=stack_base, label=column, color=color, align="center")
            stack_base += values
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(driver_labels, rotation=45, ha="right")
        ax.legend(title="Class", loc="upper right")
        ax.margins(x=0.01)
        st.pyplot(fig)


context_columns = [
    "DriverNumber",
    "FirstName",
    "LastName",
    "TeamName",
    "Year",
    "RaceName",
]
extra_columns = filtered_df[context_columns].reset_index(drop=True)
# Reconstruct display-friendly driver names that align with the feature matrix indices
driver_labels = derive_driver_labels(extra_columns)

predicted_positions: pd.Series | None = None

if position_model is not None:
    predicted_positions = pd.Series(position_model.predict(features), name="PredictedPosition")
    actual_positions = (
        filtered_df["Position"].reset_index(drop=True)
        if "Position" in filtered_df
        else pd.Series(dtype=float)
    )
    write_prediction_section(
        "Predicted finishing positions",
        predicted_positions,
        extra_columns.assign(ActualPosition=actual_positions),
    )
else:
    st.info("Train the regression model (position_regressor.pkl) to view finishing-position predictions.")

q3_probabilities = pd.DataFrame()
if q3_model is not None:
    q3_predictions = pd.Series(q3_model.predict(features), name="PredictedMadeQ3")
    q3_probabilities = pd.DataFrame(q3_model.predict_proba(features), columns=["P(No Q3)", "P(Made Q3)"])
    q3_probability_view = pd.concat(
        [pd.Series(driver_labels.tolist(), name="Driver"), q3_probabilities.reset_index(drop=True)],
        axis=1,
    ).set_index("Driver")
    actual_q3 = (
        filtered_df["y_q3"].reset_index(drop=True)
        if "y_q3" in filtered_df
        else pd.Series(dtype=float)
    )
    write_prediction_section(
        "Probability of reaching Q3",
        q3_predictions,
        extra_columns.assign(ActualMadeQ3=actual_q3),
        probability_df=q3_probability_view,
    )
else:
    st.info("Train the Q3 classifier (q3_classifier.pkl) to view make-Q3 probabilities.")

top10_probabilities = pd.DataFrame()
if top10_model is not None:
    top10_predictions = pd.Series(top10_model.predict(features), name="PredictedTop10")
    top10_probabilities = pd.DataFrame(
        top10_model.predict_proba(features),
        columns=["P(Outside Top 10)", "P(Top 10)"],
    )
    top10_probability_view = pd.concat(
        [pd.Series(driver_labels.tolist(), name="Driver"), top10_probabilities.reset_index(drop=True)],
        axis=1,
    ).set_index("Driver")
    actual_positions = (
        filtered_df["y_position"].reset_index(drop=True)
        if "y_position" in filtered_df
        else pd.Series(dtype=float)
    )
    actual_top10 = actual_positions.le(10) if actual_positions.notna().any() else pd.Series([pd.NA] * len(features))
    write_prediction_section(
        "Chance of qualifying inside the top 10",
        top10_predictions,
        extra_columns.assign(ActualTop10=actual_top10),
        probability_df=top10_probability_view,
    )
else:
    st.info("Train the Top 10 classifier (top10_classifier.pkl) to view top-10 probabilities.")


# --------- Additional insight charts ---------
st.markdown("---")
st.header("Trend highlights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Average predicted finishing position by team**")
    if predicted_positions is not None:
        team_source = pd.concat(
            [extra_columns.reset_index(drop=True), predicted_positions.reset_index(drop=True)],
            axis=1,
        ).dropna(subset=["TeamName", "PredictedPosition"])
        team_summary = team_source.groupby("TeamName")["PredictedPosition"].mean().sort_values()
        if not team_summary.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = sns.color_palette("viridis", len(team_summary))
            ax.barh(team_summary.index, team_summary.values, color=colors)
            ax.invert_yaxis()
            ax.set_xlabel("Predicted position (lower is better)")
            ax.set_ylabel("Team")
            st.pyplot(fig)
        else:
            st.info("Predicted positions are unavailable for the selected filters.")
    else:
        st.info("Train the regression model to unlock team-level trends.")

with col2:
    st.markdown("**Drivers with highest probability of making Q3**")
    if not q3_probabilities.empty:
        q3_source = pd.DataFrame(
            {
                "Driver": driver_labels.tolist(),
                "P(Made Q3)": q3_probabilities["P(Made Q3)"].astype(float),
            }
        ).dropna(subset=["Driver", "P(Made Q3)"])
        # Collapse multiple appearances per driver down to an average confidence score
        q3_prob_values = (
            q3_source.groupby("Driver")["P(Made Q3)"].mean().sort_values(ascending=False).head(10)
        )
        if not q3_prob_values.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = sns.color_palette("rocket", len(q3_prob_values))
            ax.barh(q3_prob_values.index, q3_prob_values.values, color=colors)
            ax.invert_yaxis()
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            ax.set_ylabel("Driver")
            st.pyplot(fig)
        else:
            st.info("Model probabilities are unavailable for the selected filters.")
    else:
        st.info("Train the Q3 classifier to unlock this leaderboard.")

st.success("Dashboard ready. Adjust the filters or simulate the next round to explore the predictions.")


# --------- Model insights ---------
st.markdown("---")
st.header("Model insights")

# Pull cached test-set predictions so we can surface error metrics without rerunning training
results_path, used_fallback_results = resolve_existing_path(
    RESULTS_PATH, FALLBACK_RESULTS_PATH, required=False
)
results_df = load_prediction_results(results_path)

if used_fallback_results and not results_df.empty:
    st.caption(
        "Bundled sample prediction metrics loaded. Train locally and redeploy to refresh model diagnostics."
    )

if results_df.empty:
    st.info(
        "No stored predictions are available yet. Run the training script to populate model diagnostics."
    )
else:
    # Expose per-model diagnostics in dedicated tabs so interviewers can jump to any estimator quickly
    reg_tab, q3_tab, top10_tab = st.tabs(
        ["Position Regression", "Q3 Classifier", "Top 10 Classifier"]
    )

    with reg_tab:
        regression_df = (
            results_df[results_df["target"] == "position"].copy()
            if "target" in results_df
            else pd.DataFrame()
        )
        if regression_df.empty:
            st.info("Train the regression model to view detailed error diagnostics.")
        else:
            regression_df["actual"] = pd.to_numeric(regression_df["actual"], errors="coerce")
            regression_df["predicted"] = pd.to_numeric(regression_df["predicted"], errors="coerce")
            regression_df = regression_df.dropna(subset=["actual", "predicted"])

            if regression_df.empty:
                st.info("Valid regression records were not found in the stored results.")
            else:
                reg_metrics = calculate_regression_metrics(regression_df)
                metric_display = pd.DataFrame([reg_metrics]).applymap(
                    lambda x: f"{x:.3f}" if isinstance(x, (int, float)) and pd.notna(x) else x
                )
                st.subheader("Error summary")
                st.dataframe(metric_display, width="stretch", hide_index=True)

                st.subheader("Actual vs predicted grid position")
                # Scatter the held-out predictions against true values to highlight any bias
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.scatter(regression_df["actual"], regression_df["predicted"], alpha=0.6, color="#1f77b4")
                min_bound = min(regression_df["actual"].min(), regression_df["predicted"].min())
                max_bound = max(regression_df["actual"].max(), regression_df["predicted"].max())
                ax.plot([min_bound, max_bound], [min_bound, max_bound], linestyle="--", color="gray")
                ax.set_xlabel("Actual Position")
                ax.set_ylabel("Predicted Position")
                ax.set_xlim(min_bound, max_bound)
                ax.set_ylim(min_bound, max_bound)
                ax.set_title("Regression calibration plot")
                st.pyplot(fig)

                st.subheader("Distribution of residuals")
                residuals = regression_df["predicted"] - regression_df["actual"]
                fig_resid, ax_resid = plt.subplots(figsize=(6, 4))
                # Overlay KDE to show whether errors centre on zero or skew in one direction
                sns.histplot(residuals, bins=20, ax=ax_resid, kde=True, color="#ff7f0e")
                ax_resid.axvline(0, color="black", linestyle="--", linewidth=1)
                ax_resid.set_xlabel("Prediction Error (Predicted - Actual)")
                ax_resid.set_ylabel("Count")
                st.pyplot(fig_resid)

    with q3_tab:
        q3_df = (
            results_df[results_df["target"] == "q3"].copy()
            if "target" in results_df
            else pd.DataFrame()
        )
        if q3_df.empty:
            st.info("Train the Q3 classifier to view its performance diagnostics.")
        else:
            # Normalise persisted values to integers so confusion-matrix counts are correct
            q3_df["actual"] = pd.to_numeric(q3_df["actual"], errors="coerce").round().astype("Int64")
            q3_df["predicted"] = pd.to_numeric(q3_df["predicted"], errors="coerce").round().astype("Int64")
            q3_df = q3_df.dropna(subset=["actual", "predicted"])

            if q3_df.empty:
                st.info("Valid Q3 classification records were not found in the stored results.")
            else:
                q3_metrics = calculate_classification_metrics(q3_df)
                metric_display = pd.DataFrame([q3_metrics])
                metric_display[["Accuracy", "Precision", "Recall", "F1 Score"]] = metric_display[
                    ["Accuracy", "Precision", "Recall", "F1 Score"]
                ].applymap(lambda x: f"{x:.3f}")
                st.subheader("Classification summary")
                st.dataframe(metric_display, width="stretch", hide_index=True)

                st.subheader("Confusion matrix")
                matrix = pd.crosstab(q3_df["actual"], q3_df["predicted"], dropna=False)
                matrix = matrix.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
                # Heatmap keeps the binary matrix legible while surfacing class imbalances instantly
                fig_matrix, ax_matrix = plt.subplots(figsize=(4, 4))
                sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_matrix)
                ax_matrix.set_xlabel("Predicted")
                ax_matrix.set_ylabel("Actual")
                st.pyplot(fig_matrix)

    with top10_tab:
        top10_df = (
            results_df[results_df["target"] == "top10"].copy()
            if "target" in results_df
            else pd.DataFrame()
        )
        if top10_df.empty:
            st.info("Train the Top 10 classifier to view its performance diagnostics.")
        else:
            # Apply the same cleansing so the top-10 confusion matrix matches scikit-learn metrics
            top10_df["actual"] = pd.to_numeric(top10_df["actual"], errors="coerce").round().astype("Int64")
            top10_df["predicted"] = pd.to_numeric(top10_df["predicted"], errors="coerce").round().astype("Int64")
            top10_df = top10_df.dropna(subset=["actual", "predicted"])

            if top10_df.empty:
                st.info("Valid Top 10 classification records were not found in the stored results.")
            else:
                top10_metrics = calculate_classification_metrics(top10_df)
                metric_display = pd.DataFrame([top10_metrics])
                metric_display[["Accuracy", "Precision", "Recall", "F1 Score"]] = metric_display[
                    ["Accuracy", "Precision", "Recall", "F1 Score"]
                ].applymap(lambda x: f"{x:.3f}")
                st.subheader("Classification summary")
                st.dataframe(metric_display, width="stretch", hide_index=True)

                st.subheader("Confusion matrix")
                matrix = pd.crosstab(top10_df["actual"], top10_df["predicted"], dropna=False)
                matrix = matrix.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
                # Reuse the same visual treatment to keep comparisons between classifiers straightforward
                fig_matrix, ax_matrix = plt.subplots(figsize=(4, 4))
                sns.heatmap(matrix, annot=True, fmt="d", cmap="Greens", cbar=False, ax=ax_matrix)
                ax_matrix.set_xlabel("Predicted")
                ax_matrix.set_ylabel("Actual")
                st.pyplot(fig_matrix)
