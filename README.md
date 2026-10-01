<h1 align="center">F1 Race Predictor</h1>

<p align="center">Analyse Formula 1 qualifying sessions and estimate grid positions with classical machine learning.</p>

<p align="center">
  <a href="https://f1-race-predictor-oliverperrin.streamlit.app"><strong>Live dashboard</strong></a>
</p>

<p align="center">
  <img alt="MIT license" src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" />
  <img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11+-3776ab?style=flat-square" />
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.9-f7931e?style=flat-square" />
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-dashboard-ff4b4b?style=flat-square" />
</p>

---

### What it is

A small end-to-end project: it collects qualifying and race data with FastF1, builds features, trains three models and presents the results in a Streamlit dashboard.

| Model | Predicts |
| --- | --- |
| Random forest regressor | Qualifying position |
| Logistic regression | Whether a driver reaches Q3 |
| Logistic regression | Whether a driver qualifies in the top ten |

The dashboard has three views: historic weekends, a simulation of an upcoming weekend based on each driver's recent form, and model diagnostics. The hosted dashboard sleeps when idle and takes a moment to wake.

### Results

Data was last refreshed on 22 September 2026: 1,267 qualifying entries across 62 weekends from 2024 to 2026, up to round 14, the Spanish Grand Prix. Scores are from a random 20% hold-out of 253 samples per model.

| Target | Metric | Score |
| --- | --- | --- |
| Position | Mean absolute error | 1.74 grid places |
| Position | Root mean squared error | 2.09 grid places |
| Reaches Q3 | Accuracy / precision / recall | 87.4% / 81.2% / 96.8% |
| Top ten | Accuracy / precision / recall | 88.5% / 84.0% / 96.2% |

**Limit:** the models use lap times from the qualifying session itself, so these scores describe analysis after the session. They are not a measure of how well the project forecasts a session before it happens.

### Quick start

```bash
git clone https://github.com/OliverPerrin/F1-Race-Predictor.git
cd F1-Race-Predictor
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run src/visualization.py
```

The repository includes a processed dataset and trained models, so the dashboard runs without collecting data first.

### Retrain from fresh data

```bash
python tools/update_assets.py --full
python -m unittest discover -s tests -v
```

This collects every season from 2024 to the current year, rebuilds the three models and rewrites the bundled data and evaluation results. To run the steps separately:

```bash
python src/data_collection.py   # download sessions with FastF1
python src/preprocessing.py     # build the feature table
python src/model.py             # train and evaluate
```

### Layout

```text
src/data_collection.py   Downloads race and qualifying sessions, with caching and retries
src/preprocessing.py     Merges raw data and builds features and targets
src/model.py             Trains and evaluates the three models
src/visualization.py     Streamlit dashboard
src/sample_data/         Bundled dataset and hold-out predictions
data/predictions/        Trained models
tools/update_assets.py   One-command refresh
tests/                   Dashboard and refresh checks
```

### Licence

[MIT licensed](LICENSE). Data comes from [FastF1](https://github.com/theOehrly/Fast-F1) and [Jolpica-F1](https://github.com/jolpica/jolpica-f1). Built by [Oliver Perrin](https://github.com/OliverPerrin) as a learning project in classical machine learning.
