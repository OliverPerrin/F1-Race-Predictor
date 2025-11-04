import os
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             r2_score)
from sklearn.model_selection import train_test_split


DATA_PATH = "data/processed/processed_data.csv"
PREDICTIONS_DIR = "data/predictions"


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def build_feature_matrix(df: pd.DataFrame, target_columns: List[str]) -> pd.DataFrame:
    """Keep only usable numeric features that are not targets and contain data."""
    feature_df = df.select_dtypes(include=["number"]).copy()
    feature_df = feature_df.drop(columns=target_columns + ["Position"], errors="ignore")
    feature_df = feature_df.dropna(axis=1, how="all")
    if feature_df.empty:
        raise ValueError("No numeric features available after dropping empty columns.")
    return feature_df


def train_regression(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[RandomForestRegressor, Dict[str, float], pd.Series, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    predictions = pd.Series(model.predict(X_test), index=y_test.index)
    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }
    return model, metrics, y_test, predictions


def train_classification(
    X: pd.DataFrame,
    y: pd.Series,
    label: str,
) -> Tuple[LogisticRegression, Dict[str, float], pd.Series, pd.Series]:
    if y.nunique(dropna=True) < 2:
        raise ValueError(f"Target '{label}' contains fewer than two classes.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train, y_train)
    predictions = pd.Series(model.predict(X_test), index=y_test.index)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
    }
    return model, metrics, y_test, predictions


def main() -> None:
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    # Load the preprocessed dataset that already encodes categorical features
    df = load_dataset(DATA_PATH)

    target_columns = [col for col in ["y_position", "y_q3"] if col in df.columns]
    # Mirror the training features we used during experimentation
    X = build_feature_matrix(df, target_columns)

    y_position = df["y_position"]
    # Fit the qualifying position regressor on historical data
    reg_model, reg_metrics, y_test_pos, reg_predictions = train_regression(X, y_position)
    print(
        "Regression — Finishing Position\n"
        f"MAE: {reg_metrics['mae']:.3f}, R²: {reg_metrics['r2']:.3f}"
    )
    # Persist the trained estimator so the Streamlit app can reuse it
    joblib.dump(reg_model, os.path.join(PREDICTIONS_DIR, "position_regressor.pkl"))

    results_frames = [
        pd.DataFrame(
            {
                "target": "position",
                "actual": y_test_pos.values,
                "predicted": reg_predictions.values,
            }
        )
    ]

    if "y_q3" in df.columns:
        try:
            # Train a binary classifier predicting whether a driver reaches Q3
            clf_model, clf_metrics, y_test_q3, pred_q3 = train_classification(X, df["y_q3"], "y_q3")
            print(
                "Classification — Made Q3\n"
                f"Accuracy: {clf_metrics['accuracy']:.3f}, F1: {clf_metrics['f1']:.3f}"
            )
            # Save artefacts and hold-out predictions for dashboard diagnostics
            joblib.dump(clf_model, os.path.join(PREDICTIONS_DIR, "q3_classifier.pkl"))
            results_frames.append(
                pd.DataFrame(
                    {
                        "target": "q3",
                        "actual": y_test_q3.values,
                        "predicted": pred_q3.values,
                    }
                )
            )
        except ValueError as exc:
            print(f"Classification — Made Q3\nSkipped ({exc}).")

    top10_target = (y_position <= 10).astype(int)
    if top10_target.nunique() > 1:
        # Reuse the classification pipeline to flag top-10 qualifying results
        clf_model_top10, clf_metrics_top10, y_test_top10, pred_top10 = train_classification(
            X, top10_target, "top10"
        )
        print(
            "Classification — Qualified Top 10\n"
            f"Accuracy: {clf_metrics_top10['accuracy']:.3f}, F1: {clf_metrics_top10['f1']:.3f}"
        )
        # Stash the model and evaluation outputs for later visualisation
        joblib.dump(clf_model_top10, os.path.join(PREDICTIONS_DIR, "top10_classifier.pkl"))
        results_frames.append(
            pd.DataFrame(
                {
                    "target": "top10",
                    "actual": y_test_top10.values,
                    "predicted": pred_top10.values,
                }
            )
        )

    # Collate every prediction snapshot so the Streamlit dashboard can display metrics
    pd.concat(results_frames, ignore_index=True).to_csv(
        os.path.join(PREDICTIONS_DIR, "results.csv"), index=False
    )
    print("Models trained and results saved.")


if __name__ == "__main__":
    main()
