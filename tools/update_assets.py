"""Utility to refresh data, retrain models, and bundle artefacts for Streamlit Cloud.

Steps:
1) Run data collection (optional, can be skipped if data/raw already up to date).
2) Run preprocessing to rebuild processed_data.csv.
3) Run model training to refresh pickles and results.csv.
4) Copy the processed data and prediction results into src/sample_data for cloud fallback.

Usage:
    python tools/update_assets.py --full          # run all steps
    python tools/update_assets.py --skip-collect   # skip data_collection
    python tools/update_assets.py --skip-train     # skip model training

Requirements: run locally with your FastF1 credentials/network. Streamlit Cloud should use the copied artefacts only.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = REPO_ROOT / "data" / "processed" / "processed_data.csv"
RESULTS_CSV = REPO_ROOT / "data" / "predictions" / "results.csv"
SAMPLE_DIR = REPO_ROOT / "src" / "sample_data"


def run_step(args: list[str], description: str) -> None:
    print(f"\n==> {description}")
    subprocess.run(args, check=True)


def copy_artefacts() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_PROCESSED.exists():
        shutil.copy2(DATA_PROCESSED, SAMPLE_DIR / "processed_data.csv")
        print(f"Copied {DATA_PROCESSED} -> {SAMPLE_DIR / 'processed_data.csv'}")
    else:
        print("WARNING: processed_data.csv not found; skipping copy.")

    if RESULTS_CSV.exists():
        shutil.copy2(RESULTS_CSV, SAMPLE_DIR / "results.csv")
        print(f"Copied {RESULTS_CSV} -> {SAMPLE_DIR / 'results.csv'}")
    else:
        print("WARNING: results.csv not found; skipping copy.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh data, train, and bundle artefacts")
    parser.add_argument("--skip-collect", action="store_true", help="Skip running data_collection.py")
    parser.add_argument("--skip-train", action="store_true", help="Skip running model.py")
    args = parser.parse_args()

    if not args.skip_collect:
        run_step(["python3", "src/data_collection.py"], "Collecting raw data")
    else:
        print("Skipping data collection (per flag)")

    run_step(["python3", "src/preprocessing.py"], "Preprocessing data")

    if not args.skip_train:
        run_step(["python3", "src/model.py"], "Training models")
    else:
        print("Skipping model training (per flag)")

    copy_artefacts()
    print("\nAll done. Commit and push src/sample_data/ to update Streamlit Cloud fallbacks.")


if __name__ == "__main__":
    main()
