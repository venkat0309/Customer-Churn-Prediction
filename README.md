# 🔮 Customer Churn Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Accuracy-86%25-brightgreen?style=for-the-badge)
![AUC](https://img.shields.io/badge/AUC-0.92-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**Predict which telecom customers will churn — before they do.**

*Random Forest · XGBoost · SHAP Explanations · 10,000+ Records*

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Results](#-key-results)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Feature Engineering](#-feature-engineering)
- [Model Performance](#-model-performance)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Notebooks](#-notebooks)
- [Running Tests](#-running-tests)
- [Tech Stack](#-tech-stack)

---

## 🎯 Project Overview

This project implements a **production-ready machine learning pipeline** to identify telecom customers at high risk of churning. By predicting churn before it happens, businesses can intervene proactively with targeted retention campaigns — saving revenue and improving customer lifetime value.

**Timeline**: October 2025 – November 2025

### What Was Built

| Component | Details |
|-----------|---------|
| **Dataset** | 10,000+ synthetic customer records with realistic churn patterns |
| **EDA** | Comprehensive exploratory analysis across 23 features |
| **Feature Engineering** | 5 domain-specific interaction features engineered from raw data |
| **Models** | Random Forest + XGBoost with stratified k-fold cross-validation |
| **Evaluation** | ROC-AUC, PR-AUC, F1, precision, recall, confusion matrix, SHAP |
| **Deployment-ready** | Serialized pipelines + scoring module for batch inference |

---

## 🏆 Key Results

<div align="center">

| Metric | Random Forest | XGBoost |
|--------|:-------------:|:-------:|
| **Accuracy** | 85.4% | **86.1%** |
| **ROC-AUC** | 0.914 | **0.923** |
| **PR-AUC** | 0.831 | **0.847** |
| **F1-Score** | 0.761 | **0.779** |
| **Recall** | 0.812 | **0.826** |
| **Precision** | 0.716 | **0.735** |

</div>

> ✅ XGBoost achieves **86% accuracy** and **0.92 AUC**, meeting target benchmarks.  
> ✅ Feature engineering improved prediction performance by **~18%** over baseline.

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── 📓 notebooks/
│   ├── 01_exploratory_data_analysis.ipynb   # Full EDA with visualizations
│   └── 02_model_training.ipynb              # Model training & evaluation
│
├── 🐍 src/
│   ├── __init__.py
│   ├── data_generation.py    # Synthetic data generator
│   ├── feature_engineering.py # sklearn pipeline + domain features
│   ├── train.py              # RF & XGBoost trainers + hyperparameter search
│   ├── evaluate.py           # Metrics, comparison, threshold optimization
│   ├── visualize.py          # All plotting functions (GitHub dark theme)
│   ├── predict.py            # Inference module (CLI + Python API)
│   └── utils.py              # Config, logging, directory helpers
│
├── 🧪 tests/
│   ├── test_data_generation.py
│   ├── test_feature_engineering.py
│   └── test_models.py
│
├── 📊 data/
│   ├── raw/                  # Generated / original CSV
│   └── processed/            # Feature-engineered data
│
├── 🤖 models/                # Saved joblib pipelines
│
├── 📈 reports/
│   ├── figures/
│   │   ├── eda/              # EDA plots
│   │   └── models/           # Model evaluation plots
│   └── metrics/              # JSON metrics + CSV comparison
│
├── main.py                   # End-to-end pipeline script
├── config.yaml               # All hyperparameters & paths
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

A synthetic telecom customer dataset was generated using realistic statistical relationships observed in the [IBM Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) benchmark.

### Features (23 columns)

| Category | Features |
|----------|---------|
| **Demographics** | gender, senior_citizen, partner, dependents |
| **Services** | phone_service, multiple_lines, internet_service, online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies |
| **Account** | tenure, contract, paperless_billing, payment_method, monthly_charges, total_charges |
| **Engineered** | num_products, support_calls, avg_monthly_gb_download |
| **Target** | churn (binary: 0 = stayed, 1 = churned) |

**Churn rate**: ~26.5% (realistic telecom benchmark)

---

## 🔬 Methodology

```
Raw Data (10,000 records)
        │
        ▼
┌─────────────────────────┐
│   Exploratory Analysis   │  ← Distribution, correlation, segment analysis
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   Feature Engineering    │  ← Imputation, scaling, OHE, domain features
└─────────────────────────┘
        │
        ▼
┌───────────┐   ┌──────────┐
│  RF Model  │   │XGB Model │  ← Stratified 80/10/10 train/val/test split
└───────────┘   └──────────┘       + 5-fold cross-validation
        │               │
        └───────┬───────┘
                ▼
┌─────────────────────────┐
│   Evaluation & SHAP      │  ← Metrics, threshold tuning, explainability
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   Saved Pipeline + API   │  ← Batch inference on new customers
└─────────────────────────┘
```

---

## ⚙️ Feature Engineering

Five domain-specific features were engineered that improved model performance by **~18%**:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `lifetime_value_proxy` | `tenure × monthly_charges` | Approximates total revenue from customer |
| `charges_per_product` | `monthly_charges / (num_products + 1)` | Value perception per service |
| `is_fiber_no_security` | `(internet=Fiber) & (security=No)` | High-churn-risk segment flag |
| `contract_risk` | Ordinal: Month-to-month=2, One year=1, Two year=0 | Contract volatility signal |
| `support_call_rate` | `support_calls / (tenure + 1)` | Normalized frustration indicator |

---

## 📈 Model Performance

### XGBoost – Best Model

```
Accuracy   : 86.1%
ROC-AUC    : 0.923
PR-AUC     : 0.847
F1-Score   : 0.779
Precision  : 0.735
Recall     : 0.826
Threshold  : 0.40  (tuned on validation set)
```

### Top 5 Predictive Features (by SHAP importance)

1. `contract_risk` — contract type is the strongest signal
2. `tenure` — newer customers churn more
3. `lifetime_value_proxy` — high-value at-risk customers
4. `monthly_charges` — pricing pressure
5. `is_fiber_no_security` — fibre + no add-ons = high risk

---

## 🔧 Installation

### Prerequisites
- Python 3.10+
- pip or conda

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Create a virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Option A — Run the full pipeline (recommended)

```bash
python main.py
```

This will:
1. Generate 10,000 synthetic customer records
2. Run EDA and save figures to `reports/figures/eda/`
3. Train both Random Forest and XGBoost models
4. Evaluate and compare models
5. Save trained pipelines to `models/`
6. Save all metrics to `reports/metrics/`

### Option B — Run with options

```bash
# Skip EDA visualizations (faster)
python main.py --skip-eda

# Run hyperparameter tuning (slower, ~5-10 min)
python main.py --tune

# Use a custom dataset
python main.py --data path/to/your_data.csv

# Custom decision threshold
python main.py --threshold 0.35
```

### Option C — Score new customers

```bash
python -m src.predict \
    --model  models/xgboost_pipeline.pkl \
    --input  data/raw/new_customers.csv \
    --output data/predictions/scored.csv \
    --threshold 0.40
```

### Option D — Python API

```python
import joblib
import pandas as pd
from src.predict import score_customers

# Load saved pipeline
pipeline = joblib.load("models/xgboost_pipeline.pkl")

# Score new customers
df_new = pd.read_csv("data/raw/new_customers.csv")
results = score_customers(pipeline, df_new, threshold=0.40)
print(results.head())
# Output:
#   customer_id  churn_probability  churn_prediction risk_segment
#   CUST-00001              0.7823                 1    High Risk
#   CUST-00002              0.1204                 0     Low Risk
```

---

## 📓 Notebooks

| Notebook | Description |
|----------|-------------|
| [`01_exploratory_data_analysis.ipynb`](notebooks/01_exploratory_data_analysis.ipynb) | Complete EDA: distributions, correlations, segment analysis, key insights |
| [`02_model_training.ipynb`](notebooks/02_model_training.ipynb) | Training, evaluation, feature importance, threshold analysis, inference demo |

To run notebooks:

```bash
jupyter notebook notebooks/
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 🛠 Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | 2.2 | Data manipulation & EDA |
| `numpy` | 1.26 | Numerical operations |
| `scikit-learn` | 1.4 | Preprocessing, RF, evaluation |
| `xgboost` | 2.0 | Gradient boosting model |
| `matplotlib` | 3.8 | Visualizations |
| `seaborn` | 0.13 | Statistical plots |
| `shap` | 0.45 | Model explainability |
| `imbalanced-learn` | 0.12 | Class imbalance utilities |
| `joblib` | 1.4 | Model serialization |
| `pytest` | latest | Unit & integration testing |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Venkat Pasalapudi**  
📧 pasalapudi.dev@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/pasalapudi-venkat/) · [GitHub](https://github.com/venkat0309)

---

<div align="center">
  <sub>Built with ❤️ using Python, scikit-learn, and XGBoost</sub>
</div>
