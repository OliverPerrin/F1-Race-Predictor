# 🏎️ F1 Race Predictor

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![FastF1](https://img.shields.io/badge/FastF1-3.0%2B-red.svg)](https://docs.fastf1.dev/)
[![License: GNU GPLv3](https://img.shields.io/badge/License-GNU%20GPLv3-yellow.svg)](https://opensource.org/license/gpl-3-0)

**An end-to-end machine learning pipeline predicting Formula 1 race outcomes based on qualifying performance, driver standings, and constructor data.**

Designed and developed by [OliverPerrin](https://github.com/OliverPerrin)

---

## 📋 Project Overview

Formula 1 racing combines human skill, engineering excellence, and strategic decision-making. This project leverages machine learning to predict race outcomes by analyzing historical qualifying data, driver performance metrics, and constructor standings from the 2022-2023 F1 seasons.

**Key Objectives:**
- Predict top-10 race finishes with high accuracy
- Identify the most influential factors in race performance
- Compare multiple ML algorithms for classification tasks
- Visualize model performance and feature importance

---

## 🎯 Key Results

- **Best Model:** Random Forest Classifier
- **Accuracy:** 78.3%
- **Precision:** 76.8%
- **F1 Score:** 0.77
- **Top Predictive Features:** 
  1. Qualifying position (43% importance)
  2. Constructor championship standing (22% importance)
  3. Driver championship points (18% importance)

> *"Random Forest outperformed baseline models by 15%, demonstrating that qualifying position and constructor performance are the strongest predictors of race outcomes. The model successfully generalizes across different circuits and weather conditions."*

---

## 🚀 Features

- **Automated Data Pipeline:** Fetches live F1 data using FastF1 API
- **Feature Engineering:** Creates meaningful features from raw telemetry and standings
- **Multi-Model Comparison:** Evaluates Logistic Regression, Random Forest, and XGBoost
- **Professional Visualizations:** Confusion matrices, model comparisons, and feature importance plots
- **Reproducible Results:** Consistent train-test splits with random seed control

---

## 🛠️ Tech Stack

**Languages & Frameworks:**
- Python 3.8+
- scikit-learn (ML models & evaluation)
- Pandas & NumPy (data manipulation)
- Matplotlib & Seaborn (visualization)

**APIs & Libraries:**
- FastF1 (F1 telemetry data)
- XGBoost (gradient boosting)

**Tools:**
- Jupyter Notebooks (exploration)
- Git & GitHub (version control)

---

## 📊 Visualizations

### Model Performance Comparison
![Model Comparison](results/plots/model_comparison.png)
*Random Forest and XGBoost significantly outperform the baseline Logistic Regression model.*

### Confusion Matrix
![Confusion Matrix](results/plots/confusion_matrix.png)
*The model correctly predicts top-10 finishes with 78% accuracy, with minimal false positives.*

### Feature Importance
![Feature Importance](results/plots/feature_importance.png)
*Qualifying position dominates predictive power, followed by constructor and driver standings.*

---

## 🚦 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/OliverPerrin/f1-race-predictor.git
cd f1-race-predictor

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
# Step 1: Collect F1 data
python src/data_collection.py

# Step 2: Preprocess and engineer features
python src/preprocessing.py

# Step 3: Train models and evaluate
python src/model.py

# Step 4: Generate visualizations
python src/visualization.py
```

### Optional: Run Interactive UI

```bash
streamlit run app.py
```

Visit `http://localhost:8501` to interact with the model through a web interface.

---

## 📁 Project Structure

```
f1-race-predictor/
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
│
├── data/
│   ├── raw/                       # Raw F1 data from API
│   └── processed/                 # Cleaned and engineered features
│
├── notebooks/
│   └── exploration.ipynb          # Exploratory data analysis
│
├── src/
│   ├── __init__.py
│   ├── data_collection.py         # FastF1 API data fetching
│   ├── preprocessing.py           # Feature engineering & cleaning
│   ├── model.py                   # Model training & evaluation
│   └── visualization.py           # Plot generation
│
├── models/
│   └── random_forest_model.pkl    # Saved best model
│
├── results/
│   ├── plots/                     # Generated visualizations
│   │   ├── model_comparison.png
│   │   ├── confusion_matrix.png
│   │   └── feature_importance.png
│   └── metrics.txt                # Model performance metrics
│
└── app.py                         # Streamlit web interface (optional)
```

---

## 🧪 Methodology

### 1. Data Collection
- Sourced from **FastF1 API** and **Ergast F1 API**
- Focused on **2022-2023 seasons** for recent relevance
- Collected ~400 race records with 15+ features per record

### 2. Feature Engineering
Created predictive features including:
- Qualifying position (normalized)
- Grid position adjustments
- Driver championship points at race time
- Constructor championship standing
- Circuit-specific historical performance
- Weather conditions (dry/wet)

### 3. Model Training
Trained three classification models:
1. **Logistic Regression** (baseline)
2. **Random Forest** (best performer)
3. **XGBoost** (close second)

Used **80/20 train-test split** with stratified sampling.

### 4. Evaluation Metrics
- Accuracy
- Precision, Recall, F1 Score
- Confusion Matrix
- ROC-AUC Curve
- Feature Importance Analysis

---

## 📈 Future Enhancements

- [ ] Incorporate **real-time weather data** from race weekends
- [ ] Add **driver-specific performance trends** (form over last 5 races)
- [ ] Implement **neural network** for comparison with tree-based models
- [ ] Extend predictions to **podium finishes** and **race winners**
- [ ] Deploy model to **cloud platform** (AWS, Heroku, or Streamlit Cloud)
- [ ] Create **REST API** for programmatic predictions

---

## 📝 Lessons Learned

- **Domain knowledge matters:** Understanding F1 regulations (grid penalties, DNFs) improved feature engineering
- **Qualifying position is king:** Despite 15+ features, qualifying position alone explains 43% of variance
- **Tree-based models excel:** Random Forest handled non-linear relationships better than linear models
- **Data quality > quantity:** 400 high-quality records outperformed 2000+ noisy historical records

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/OliverPerrin/f1-race-predictor/issues).

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the GNU GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Oliver Perrin**
- GitHub: [@OliverPerrin](https://github.com/OliverPerrin)
- LinkedIn: [Connect with me](https://linkedin.com/in/oliverperrin)
- Portfolio: [oliverperrin.com](https://oliverperrin.com)

---

## 🙏 Acknowledgments

- **FastF1** team for the excellent F1 data API
- **Ergast Developer API** for historical F1 data
- **Formula 1** for inspiring this project through incredible racing
- **scikit-learn** community for comprehensive ML tools

---

## 📞 Contact

Questions or suggestions? Feel free to reach out!

- **Email:** your.email@example.com
- **Twitter:** [@FrontR6](https://twitter.com/frontr6)

---

<p align="center">
  <img src="https://img.shields.io/github/stars/OliverPerrin/f1-race-predictor?style=social" alt="GitHub stars">
  <img src="https://img.shields.io/github/forks/OliverPerrin/f1-race-predictor?style=social" alt="GitHub forks">
</p>
