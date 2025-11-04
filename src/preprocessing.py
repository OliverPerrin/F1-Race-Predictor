
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

quali_df = pd.read_csv('data/raw/qualifying_results.csv')
drop_cols = [
    "BroadcastName", "FullName", "HeadshotUrl", "TeamColor",
    "TeamId", "DriverId", "Status", "Points", "Laps"
]
quali_df = quali_df.drop(columns=[c for c in drop_cols if c in quali_df.columns])

def time_to_seconds(time_str):
    if pd.isna(time_str):
        return np.nan
    try:
        return pd.to_timedelta(time_str).total_seconds()
    except (ValueError, TypeError):
        try:
            return pd.to_timedelta(f"00:{time_str}").total_seconds()
        except (ValueError, TypeError):
            return np.nan

for col in ["Q1", "Q2", "Q3"]:
    quali_df[col] = quali_df[col].apply(time_to_seconds)

# Capture whether the driver reached Q3 before filling missing times so the label stays binary.
made_q3 = quali_df["Q3"].notna().astype(int)

# Remember original string columns that we want to keep for reporting/visualization.
original_columns = {
    "Abbreviation": quali_df.get("Abbreviation"),
    "TeamName": quali_df.get("TeamName"),
    "Country": quali_df.get("Country"),
    "Location": quali_df.get("Location"),
}

quali_df = quali_df.fillna(quali_df.mean(numeric_only=True))

quali_df["BestLap"] = quali_df[["Q1", "Q2", "Q3"]].min(axis=1)
quali_df["AvgLap"] = quali_df[["Q1", "Q2", "Q3"]].mean(axis=1)
quali_df["Improvement"] = quali_df["Q1"] - quali_df["Q3"]

label_encoders = {}
for col in ["Abbreviation", "TeamName", "Country", "Location"]:
    if col in quali_df.columns:
        le = LabelEncoder()
        encoded = le.fit_transform(quali_df[col].astype(str))
        quali_df[f"{col}_encoded"] = encoded
        label_encoders[col] = le

# Restore the original string values to their columns for readability downstream.
for col, values in original_columns.items():
    if values is not None:
        quali_df[col] = values

scaler = StandardScaler()
numeric_cols = [
    col for col in ["Q1", "Q2", "Q3", "BestLap", "AvgLap", "Improvement"]
    if col in quali_df.columns
]
quali_df[numeric_cols] = scaler.fit_transform(quali_df[numeric_cols])

quali_df["y_position"] = quali_df["Position"]
quali_df["y_q3"] = made_q3

features = [c for c in quali_df.columns if not c.startswith("y_") and c != "Position"]

X_train, X_test, y_train, y_test = train_test_split(
    quali_df[features], quali_df[["y_position", "y_q3"]], test_size=0.2, random_state=42
)

print("Shape:", quali_df.shape)
print("Example rows:\n", quali_df.head())

quali_df.to_csv("data/processed/processed_data.csv", index=False)
