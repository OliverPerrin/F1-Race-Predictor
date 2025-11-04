# F1 Race Predictor

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-blue?logo=streamlit)]([https://your-app-name.streamlit.app](https://f1-race-predictor-oliverperrin.streamlit.app))

An end-to-end workflow for analysing Formula 1 qualifying sessions, training machine-learning models, and presenting the results through an interactive Streamlit dashboard. The project pulls recent race data with FastF1, prepares a modelling dataset, trains regressors/classifiers, and surfaces the outputs in a way that is easy to demo.

## Project Overview
- Processes qualifying and race results for recent F1 seasons using FastF1 and Ergast data.
- Builds a clean feature set covering lap times, session deltas, team and driver context, and engineered targets.
- Trains three models:
  - Random Forest regressor to estimate finishing position.
  - Logistic regression to predict who reaches Q3.
  - Logistic regression to predict top-ten qualifiers.
- Ships a Streamlit application (`src/visualization.py`) for slicing historic weekends, simulating upcoming rounds, and reviewing model diagnostics.

## Highlights
- **Scenario planning:** simulate an upcoming weekend by averaging each driver's recent form and generating fresh predictions.
- **Model insights:** review scatter plots, residual distributions, and confusion matrices directly inside the dashboard.
- **Offline artefacts:** every training run serialises the models and stores test-set predictions so the analytics can be shared without re-training.

## Key Results
Metrics below are calculated from the latest `data/predictions/results.csv` artefact (176 test-set samples).

| Target | Metric | Score |
| ------ | ------ | ----- |
| Position regression | Mean absolute error | 1.78 grid places |
| Position regression | Root mean squared error | 2.28 grid places |
| Position regression | Mean bias (predicted - actual) | -0.10 |
| Q3 classification | Accuracy | 93.8% |
| Q3 classification | Precision | 88.2% |
| Q3 classification | Recall | 100% |
| Q3 classification | F1 score | 93.7% |
| Top-ten classification | Accuracy | 92.6% |
| Top-ten classification | Precision | 88.2% |
| Top-ten classification | Recall | 97.6% |
| Top-ten classification | F1 score | 92.7% |

## Tech Stack
- **Python 3.11**
- **Data & ML:** pandas, numpy, scikit-learn, joblib
- **Data ingestion:** FastF1, Ergast (via `requests`)
- **Visualisation & UI:** Streamlit, seaborn, matplotlib

## Getting Started
### Prerequisites
- Python 3.10+ (3.12 recommended)
- pip
- (Optional) virtual environment tool such as `venv` or `conda`

### Installation
```bash
# Clone the project

cd F1-Race-Predictor

# Create and activate a virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Typical Workflow
1. **Collect fresh data** (optional if `data/raw` already exists):
   ```bash
   python src/data_collection.py
   ```
2. **Preprocess and engineer features:**
   ```bash
   python src/preprocessing.py
   ```
3. **Train the models and generate metrics:**
   ```bash
   python src/model.py
   ```
   - Saves trained models to `data/predictions/`.
   - Writes hold-out predictions to `data/predictions/results.csv` for later analysis.
4. **Launch the dashboard:** see next section.

## Using the Streamlit Dashboard
```bash
streamlit run src/visualization.py
```
The app includes three main areas:
- **Historic weekends:** filter by season and race, review predicted finishing order, and inspect probabilities for Q3 / top-ten appearances.
- **Upcoming weekend simulation:** specify a future race, choose a look-back window, and the app will estimate results based on each driver's recent form.
- **Model insights tab:** compare actual vs predicted positions, explore residual distributions, and inspect confusion matrices for the classifiers.

## Repository Components
- `src/data_collection.py` – pulls race and qualifying sessions via FastF1 with retry logic and caching.
- `src/preprocessing.py` – merges raw data, converts time columns, creates engineered targets, and outputs `data/processed/processed_data.csv`.
- `src/model.py` – trains the regression/classification models and stores artefacts/metrics.
- `src/visualization.py` – Streamlit dashboard powering the interactive experience.
- `data/` – raw downloads, processed dataset, and prediction artefacts.
- `results/` – generated metrics or additional analyses.

## Contributing
Issues, ideas, and pull requests are welcome. Please open a discussion or PR if you would like to extend the models, add new visualisations, or integrate additional data sources.

## License
This project is distributed under the terms of the GNU GPL-3.0 license. See `LICENSE` for details.

## Contact
Created by [Oliver Perrin](https://github.com/OliverPerrin). For questions or collaboration, feel free to reach out on GitHub or LinkedIn.

