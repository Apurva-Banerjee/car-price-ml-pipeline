# 🚗 Car Price Prediction — ML Pipeline

> **`pipeline.py`** · CarDekho Dataset · Random Forest Regression · scikit-learn

---

## Table of Contents

1. [Overview](#1-overview)
2. [Dataset](#2-dataset)
3. [Pipeline Architecture](#3-pipeline-architecture)
4. [How It Works](#4-how-it-works)
5. [Installation](#5-installation)
6. [Usage](#6-usage)
7. [Results](#7-results)
8. [Design Decisions](#8-design-decisions)
9. [Extending the Pipeline](#9-extending-the-pipeline)

---

## 1. Overview

`pipeline.py` is a self-contained, end-to-end machine learning pipeline for predicting used car selling prices using the **CarDekho dataset**. It covers every stage of the ML lifecycle:

```
Raw CSV  →  Preprocessing  →  Baseline Comparison  →  HPO  →  Final Model  →  Artifacts
```

The pipeline is fully driven from the command line — one command reproduces all results, tunes models, and saves ready-to-use artifacts.

---

## 2. Dataset

| Property | Value |
|---|---|
| File | `cardekho_imputated.csv` |
| Rows | 15,411 |
| Columns | 13 |
| Missing values | 0 |
| Target | `selling_price` (INR) |

### Columns

| Column | Type | Description |
|---|---|---|
| `car_name` | String | Full car name — **dropped** during preprocessing |
| `brand` | String | Manufacturer brand — **dropped** during preprocessing |
| `model` | String | Car model (120 unique values) — **label-encoded** |
| `vehicle_age` | Numeric | Age of the vehicle in years |
| `km_driven` | Numeric | Total kilometres driven |
| `seller_type` | String | Individual / Dealer / Trustmark Dealer — **one-hot encoded** |
| `fuel_type` | String | Petrol / Diesel / CNG / LPG / Electric — **one-hot encoded** |
| `transmission_type` | String | Manual / Automatic — **one-hot encoded** |
| `mileage` | Numeric | Fuel efficiency (km/l or km/kg) |
| `engine` | Numeric | Engine displacement (CC) |
| `max_power` | Numeric | Maximum power output (BHP) |
| `seats` | Numeric | Number of seats |
| `selling_price` | Numeric | **TARGET** — Selling price in INR |

---

## 3. Pipeline Architecture

The pipeline is split into **9 clearly separated, independently callable stages**:

| # | Function | What It Does |
|---|---|---|
| 1 | `load_data()` | Reads the CSV, logs shape and null value counts |
| 2 | `preprocess()` | Drops columns, encodes categoricals, scales numerics, splits 80/20 |
| 3 | `evaluate_model()` | Reusable helper — computes and logs MAE, RMSE, R² |
| 4 | `run_baseline()` | Trains 6 baseline models, prints a ranked summary table |
| 5 | `tune_models()` | Runs `RandomizedSearchCV` for KNN and Random Forest *(optional)* |
| 6 | `build_final_models()` | Instantiates final models with tuned or default hyperparameters |
| 7 | `train_final()` | Trains and evaluates final models on train and test sets |
| 8 | `save_artifacts()` | Pickles trained models and the fitted preprocessor to disk |
| 9 | `predict()` | Inference helper — loads saved artifacts and scores new data |

---

## 4. How It Works

### 4.1 Data Ingestion

`load_data()` reads the CSV, logs the dataset shape, and reports missing values per column. All 15,411 rows are complete with zero nulls.

---

### 4.2 Preprocessing

`preprocess()` applies transformations in this order:

1. **Drop** `car_name` and `brand` — high-cardinality, not predictive after `model` is encoded
2. **Label-encode** `model` — 120 unique values mapped to integers 0–119
3. **One-hot encode** `seller_type`, `fuel_type`, `transmission_type` — `drop='first'` to avoid multicollinearity
4. **Standard-scale** all numeric columns — zero mean, unit variance
5. **Train / test split** — 80% training (12,328 rows) / 20% testing (3,083 rows), `random_state=42`

> ⚠️ The `ColumnTransformer` is **fit only on training data** — `transform()` is called separately for the test set to prevent data leakage.

---

### 4.3 Baseline Model Comparison

`run_baseline()` trains six models and prints a summary table ranked by Test R²:

| Model | Test R² | Test RMSE | Notes |
|---|---|---|---|
| **Random Forest** | **0.930** | ~229K | Best overall |
| KNN Regressor | 0.912 | ~258K | Strong non-linear learner |
| Decision Tree | 0.877 | ~304K | Overfits (Train R²=0.9995) |
| Ridge Regression | 0.665 | ~503K | Underfits — linear model |
| Lasso Regression | 0.665 | ~503K | Same as Ridge at default alpha |
| Linear Regression | 0.665 | ~503K | Baseline linear model |

---

### 4.4 Hyperparameter Tuning *(optional)*

Activated with `--tune`. Uses `RandomizedSearchCV` with `cv=3` over the following search spaces:

**Random Forest**
```
n_estimators      : [100, 200, 500, 1000]
max_depth         : [5, 8, 10, 15, None]
max_features      : [5, 7, 8, 'sqrt']
min_samples_split : [2, 8, 15, 20]
```

**KNN**
```
n_neighbors : [2, 3, 10, 20, 40, 50]
```

Best params found with `--n-iter 100`:
- **RF** → `n_estimators=100, min_samples_split=2, max_features=8, max_depth=None`
- **KNN** → `n_neighbors=10`

---

### 4.5 Final Evaluation

| Metric | Without Tuning | With Tuning (`n_iter=100`) |
|---|---|---|
| Test R² | 0.9440 | 0.9360 |
| Test RMSE | 205,231 | 219,528 |
| Test MAE | 98,990 | 99,688 |

---

### 4.6 Model Persistence

`save_artifacts()` pickles every trained model and the fitted preprocessor:

```
artifacts/
  ├── random_forest_regressor.pkl
  ├── k-neighbors_regressor.pkl
  └── preprocessor.pkl
```

---

### 4.7 Inference

Load saved artifacts and score new raw DataFrames with `predict()`:

```python
from pipeline import predict

predictions = predict(
    model_path="artifacts/random_forest_regressor.pkl",
    preprocessor_path="artifacts/preprocessor.pkl",
    raw_df=new_cars_df   # same columns as training data, no target column
)
```

---

## 5. Installation

**Requirements:** Python 3.8+

```bash
pip install pandas numpy scikit-learn
```

**Directory structure:**

```
project/
  ├── pipeline.py
  ├── data/
  │   └── cardekho_imputated.csv
  └── artifacts/          ← created automatically on first run
```

---

## 6. Usage

```bash
# Fast run — baseline + final RF with default params (~5 seconds)
python pipeline.py --data ./data/cardekho_imputated.csv

# Enable hyperparameter tuning (default 50 iterations)
python pipeline.py --data ./data/cardekho_imputated.csv --tune

# Full tuning with 100 random search iterations (~90 seconds)
python pipeline.py --data ./data/cardekho_imputated.csv --tune --n-iter 100

# Save artifacts to a custom directory
python pipeline.py --data ./data/cardekho_imputated.csv --out-dir models/
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--data` | `./data/cardekho_imputated.csv` | Path to the input CSV file |
| `--tune` | `False` (flag) | Enable `RandomizedSearchCV` hyperparameter tuning |
| `--n-iter` | `50` | Number of random search iterations for HPO |
| `--out-dir` | `artifacts` | Output directory for saved model artifacts |

---

## 7. Results

Sample terminal output from a full tuning run:

```
2026-04-08 14:01:40  INFO   Baseline summary (sorted by Test R²):
                                   mae           rmse        r2
Random Forest Regressor  101860.84  229578.29  0.9300
K-Neighbors Regressor    112934.07  257765.16  0.9117
Decision Tree            125447.66  310177.80  0.8722
Ridge                    279557.45  502534.47  0.6645
Lasso                    279614.76  502542.74  0.6645
Linear Regression        279618.58  502543.59  0.6645

2026-04-08 14:03:13  INFO   ── Random Forest Regressor (tuned) ──
2026-04-08 14:03:13  INFO   [TRAIN]  MAE=39315.69  RMSE=120600.93  R²=0.9821
2026-04-08 14:03:13  INFO   [TEST ]  MAE=99687.77  RMSE=219528.31  R²=0.9360
```

---

## 8. Design Decisions

- **No data leakage** — `ColumnTransformer` is fit exclusively on training data; `transform()` is called separately for the test set
- **`max_features='sqrt'` instead of `'auto'`** — `'auto'` was deprecated and removed in scikit-learn 1.2
- **`sparse_output=False`** — `OneHotEncoder` uses dense output to stay compatible with `StandardScaler` in the same `ColumnTransformer`
- **Tuning is opt-in** — HPO takes ~90s; skipping it still produces a strong model for rapid iteration
- **Single-file design** — all stages in one file with no external config, easy to read and extend
- **Structured logging** — timestamped `INFO`-level logs trace every step for reproducibility and debugging

---

## 9. Extending the Pipeline

- **Add a new model** — insert it into `BASELINE_MODELS` dict and optionally into `HPO_PARAM_GRIDS`
- **Add new features** — extend `preprocess()`; the `ColumnTransformer` handles mixed column types cleanly
- **Cross-validation scoring** — replace `train_final()` with a `cross_val_score` loop for more robust evaluation
- **MLflow tracking** — wrap `run_baseline()` and `train_final()` with `mlflow.start_run()` to log metrics automatically
- **GradientBoosting / XGBoost** — drop-in replacements for Random Forest with often better performance

---

*pipeline.py · Car Price Prediction · Built with scikit-learn*
