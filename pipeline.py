"""
pipeline.py
===========
End-to-end ML pipeline for Car Price Prediction (CarDekho dataset).
Covers: ingestion → preprocessing → baseline training → hyperparameter
tuning → final evaluation → model persistence.

Usage
-----
    python pipeline.py --data ./data/cardekho_imputated.csv
    python pipeline.py --data ./data/cardekho_imputated.csv --tune   # with HPO
"""

import argparse
import logging
import warnings
import os
import pickle

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. DATA INGESTION
# ─────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Load CSV and return a DataFrame."""
    log.info("Loading data from: %s", path)
    df = pd.read_csv(path, index_col=0)
    log.info("Dataset shape: %s", df.shape)
    log.info("Null values per column:\n%s", df.isnull().sum().to_string())
    return df


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    """
    Clean, encode, and scale features.

    Returns
    -------
    X_train, X_test, y_train, y_test, preprocessor
    """
    log.info("Starting preprocessing …")

    # Drop high-cardinality / leaky columns
    df = df.drop(columns=["car_name", "brand"], errors="ignore")
    log.info("Dropped 'car_name' and 'brand' columns.")

    # Split features / target
    X = df.drop("selling_price", axis=1)
    y = df["selling_price"]

    # Label-encode high-cardinality categorical column
    le = LabelEncoder()
    X["model"] = le.fit_transform(X["model"])
    log.info("Label-encoded 'model' column (%d unique values).", len(le.classes_))

    # Column groups
    onehot_cols = ["seller_type", "fuel_type", "transmission_type"]
    num_cols = X.select_dtypes(exclude="object").columns.tolist()

    log.info("One-hot columns : %s", onehot_cols)
    log.info("Numeric columns : %s", num_cols)

    # Build preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(drop="first", sparse_output=False), onehot_cols),
            ("scaler", StandardScaler(), num_cols),
        ],
        remainder="passthrough",
    )

    # Train / test split (80/20)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    log.info("Train shape: %s | Test shape: %s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test, preprocessor


# ─────────────────────────────────────────────
# 3. EVALUATION HELPER
# ─────────────────────────────────────────────

def evaluate_model(y_true, y_pred, split: str = ""):
    """Print and return MAE, RMSE, R²."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    tag  = f"[{split}]" if split else ""
    log.info("%s  MAE=%.4f  RMSE=%.4f  R²=%.4f", tag, mae, rmse, r2)
    return mae, rmse, r2


# ─────────────────────────────────────────────
# 4. BASELINE MODEL COMPARISON
# ─────────────────────────────────────────────

BASELINE_MODELS = {
    "Linear Regression":      LinearRegression(),
    "Lasso":                  Lasso(),
    "Ridge":                  Ridge(),
    "K-Neighbors Regressor":  KNeighborsRegressor(),
    "Decision Tree":          DecisionTreeRegressor(),
    "Random Forest Regressor":RandomForestRegressor(),
}


def run_baseline(X_train, X_test, y_train, y_test) -> dict:
    """Train all baseline models and return their test metrics."""
    log.info("=" * 60)
    log.info("BASELINE MODEL COMPARISON")
    log.info("=" * 60)
    results = {}

    for name, model in BASELINE_MODELS.items():
        model.fit(X_train, y_train)

        log.info("── %s ──", name)
        evaluate_model(y_train, model.predict(X_train), "TRAIN")
        mae, rmse, r2 = evaluate_model(y_test,  model.predict(X_test),  "TEST ")

        results[name] = {"mae": mae, "rmse": rmse, "r2": r2}

    # Summary table
    summary = pd.DataFrame(results).T.sort_values("r2", ascending=False)
    log.info("\nBaseline summary (sorted by Test R²):\n%s", summary.to_string())
    return results


# ─────────────────────────────────────────────
# 5. HYPERPARAMETER TUNING
# ─────────────────────────────────────────────

HPO_PARAM_GRIDS = {
    "KNN": (
        KNeighborsRegressor(),
        {"n_neighbors": [2, 3, 10, 20, 40, 50]},
    ),
    "RF": (
        RandomForestRegressor(),
        {
            "max_depth":         [5, 8, 15, None, 10],
            "max_features":      [5, 7, 8, "sqrt"],
            "min_samples_split": [2, 8, 15, 20],
            "n_estimators":      [100, 200, 500, 1000],
        },
    ),
}


def tune_models(X_train, y_train, n_iter: int = 50) -> dict:
    """
    Run RandomizedSearchCV for KNN and Random Forest.
    Returns dict of best params per model.
    """
    log.info("=" * 60)
    log.info("HYPERPARAMETER TUNING  (n_iter=%d, cv=3)", n_iter)
    log.info("=" * 60)
    best_params = {}

    for name, (model, param_grid) in HPO_PARAM_GRIDS.items():
        log.info("Tuning %s …", name)
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=3,
            verbose=0,
            n_jobs=-1,
            random_state=42,
        )
        search.fit(X_train, y_train)
        best_params[name] = search.best_params_
        log.info("Best params for %s: %s", name, search.best_params_)

    return best_params


# ─────────────────────────────────────────────
# 6. FINAL MODEL TRAINING
# ─────────────────────────────────────────────

def build_final_models(best_params: dict) -> dict:
    """
    Instantiate final models using tuned hyperparameters.
    Falls back to sensible defaults when tuning was skipped.
    """
    rf_params  = best_params.get("RF",  {})
    knn_params = best_params.get("KNN", {})

    return {
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators     = rf_params.get("n_estimators",      100),
            max_depth        = rf_params.get("max_depth",         None),
            max_features     = rf_params.get("max_features",      "sqrt"),
            min_samples_split= rf_params.get("min_samples_split", 2),
            n_jobs=-1,
        ),
        "K-Neighbors Regressor": KNeighborsRegressor(
            n_neighbors = knn_params.get("n_neighbors", 10),
            n_jobs=-1,
        ),
    }


def train_final(X_train, X_test, y_train, y_test, best_params: dict) -> dict:
    """Train final (tuned) models and evaluate."""
    log.info("=" * 60)
    log.info("FINAL MODEL EVALUATION")
    log.info("=" * 60)
    final_models = build_final_models(best_params)
    trained = {}

    for name, model in final_models.items():
        model.fit(X_train, y_train)
        log.info("── %s (tuned) ──", name)
        evaluate_model(y_train, model.predict(X_train), "TRAIN")
        evaluate_model(y_test,  model.predict(X_test),  "TEST ")
        trained[name] = model

    return trained


# ─────────────────────────────────────────────
# 7. MODEL PERSISTENCE
# ─────────────────────────────────────────────

def save_artifacts(trained_models: dict, preprocessor, out_dir: str = "artifacts"):
    """Pickle trained models and the preprocessor."""
    os.makedirs(out_dir, exist_ok=True)

    for name, model in trained_models.items():
        fname = name.lower().replace(" ", "_") + ".pkl"
        path  = os.path.join(out_dir, fname)
        with open(path, "wb") as f:
            pickle.dump(model, f)
        log.info("Saved model → %s", path)

    prep_path = os.path.join(out_dir, "preprocessor.pkl")
    with open(prep_path, "wb") as f:
        pickle.dump(preprocessor, f)
    log.info("Saved preprocessor → %s", prep_path)


# ─────────────────────────────────────────────
# 8. INFERENCE HELPER
# ─────────────────────────────────────────────

def predict(model_path: str, preprocessor_path: str, raw_df: pd.DataFrame) -> np.ndarray:
    """
    Load a saved model + preprocessor and return predictions.

    Parameters
    ----------
    model_path        : path to the pickled model
    preprocessor_path : path to the pickled ColumnTransformer
    raw_df            : DataFrame with the same columns as training (no target)
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)

    X = preprocessor.transform(raw_df)
    return model.predict(X)


# ─────────────────────────────────────────────
# 9. MAIN ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Car Price ML Pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default="./data/cardekho_imputated.csv",
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        default=False,
        help="Run RandomizedSearchCV hyperparameter tuning (slower).",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=50,
        help="Number of iterations for RandomizedSearchCV (default: 50).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts",
        help="Directory to save model artifacts (default: artifacts/).",
    )
    args = parser.parse_args()

    # Step 1: Load
    df = load_data(args.data)

    # Step 2: Preprocess
    X_train, X_test, y_train, y_test, preprocessor = preprocess(df)

    # Step 3: Baseline comparison
    run_baseline(X_train, X_test, y_train, y_test)

    # Step 4: Hyperparameter tuning (optional)
    best_params: dict = {}
    if args.tune:
        best_params = tune_models(X_train, y_train, n_iter=args.n_iter)
    else:
        log.info("Skipping HPO (pass --tune to enable).  Using default params.")

    # Step 5: Final training
    trained_models = train_final(X_train, X_test, y_train, y_test, best_params)

    # Step 6: Save artifacts
    save_artifacts(trained_models, preprocessor, out_dir=args.out_dir)

    log.info("Pipeline complete. Artifacts saved to '%s/'.", args.out_dir)


if __name__ == "__main__":
    main()
