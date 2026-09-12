"""
Model training, TimeSeriesSplit cross-validation, hyperparameter evaluation, and serialization.
Compares:
- Random Forest Regressor
- Support Vector Regression (SVR)
- Gradient Boosting Regressor
"""
import sys
import json
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler

from src.config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    RF_MODEL_PATH,
    SVR_MODEL_PATH,
    GB_MODEL_PATH,
    BEST_MODEL_PATH,
    SCALER_PATH,
    METRICS_PATH
)
from src.feature_engineering import run_feature_engineering_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def calculate_directional_accuracy(actual_prev: np.ndarray, actual_future: np.ndarray, pred_future: np.ndarray) -> float:
    """Calculate the percentage of times the model correctly predicts market direction (Up/Down)."""
    actual_dir = actual_future > actual_prev
    pred_dir = pred_future > actual_prev
    return float(np.mean(actual_dir == pred_dir) * 100.0)


def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray, y_prev: np.ndarray) -> dict:
    """Compute comprehensive regression and trading metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
    dir_acc = calculate_directional_accuracy(y_prev, y_true, y_pred)

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "MAPE_percent": round(mape, 4),
        "Directional_Accuracy_percent": round(dir_acc, 2)
    }


def train_and_compare_models():
    """Main training routine with TimeSeries cross-validation."""
    logging.info("Starting ML Training Pipeline...")

    df = run_feature_engineering_pipeline(save_csv=True)
    n_samples = len(df)
    logging.info(f"Loaded engineered dataset with {n_samples} samples and {len(FEATURE_COLUMNS)} features.")

    # Chronological Train-Test Split (80% Train, 20% Test) to respect time causality
    split_idx = int(n_samples * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train_raw = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN].values
    X_test_raw = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].values
    y_test_prev = test_df["close"].values

    # Fit scaler strictly on training set (Leakage Prevention)
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Initialize candidate algorithms
    models = {
        "Random Forest Regressor": {
            "model": RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_split=5, random_state=42, n_jobs=-1),
            "save_path": RF_MODEL_PATH
        },
        "Support Vector Regression (SVR)": {
            "model": SVR(C=100.0, epsilon=0.1, kernel="rbf", gamma="scale"),
            "save_path": SVR_MODEL_PATH
        },
        "Gradient Boosting Regressor": {
            "model": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42),
            "save_path": GB_MODEL_PATH
        }
    }

    results = {}
    best_model_name = None
    best_rmse = float("inf")

    # TimeSeriesSplit Cross Validation (5 folds) on Training Set
    tscv = TimeSeriesSplit(n_splits=5)

    print("\n" + "="*70)
    print("TIME-SERIES CROSS-VALIDATION & TEST SET EVALUATION")
    print("="*70)

    for name, item in models.items():
        logging.info(f"Training and validating {name}...")
        model = item["model"]

        # 5-fold TimeSeries CV
        cv_rmse_scores = []
        for train_fold_idx, val_fold_idx in tscv.split(X_train):
            X_cv_train, X_cv_val = X_train[train_fold_idx], X_train[val_fold_idx]
            y_cv_train, y_cv_val = y_train[train_fold_idx], y_train[val_fold_idx]
            model.fit(X_cv_train, y_cv_train)
            pred_val = model.predict(X_cv_val)
            cv_rmse_scores.append(np.sqrt(mean_squared_error(y_cv_val, pred_val)))

        cv_rmse_mean = float(np.mean(cv_rmse_scores))

        # Train on full train set and evaluate on unseen holdout test set
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        test_metrics = evaluate_model_performance(y_test, y_pred_test, y_test_prev)

        # Save individual model
        joblib.dump(model, item["save_path"])
        logging.info(f"Saved {name} to {item['save_path']}")

        # Extract feature importances if available (e.g. Random Forest, Gradient Boosting)
        feature_importance = {}
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_importance = {
                FEATURE_COLUMNS[i]: round(float(importances[i]), 4)
                for i in np.argsort(importances)[::-1]
            }

        results[name] = {
            "CV_RMSE_Mean": round(cv_rmse_mean, 4),
            "Test_Metrics": test_metrics,
            "Feature_Importance": feature_importance
        }

        print(f"\nModel: {name}")
        print(f"  - 5-Fold TimeSeries CV RMSE: {cv_rmse_mean:.4f}")
        print(f"  - Test MAE:                  {test_metrics['MAE']:.4f}")
        print(f"  - Test RMSE:                 {test_metrics['RMSE']:.4f}")
        print(f"  - Test R2 Score:             {test_metrics['R2']:.4f}")
        print(f"  - Test Directional Accuracy: {test_metrics['Directional_Accuracy_percent']}%")

        if test_metrics["RMSE"] < best_rmse:
            best_rmse = test_metrics["RMSE"]
            best_model_name = name

    # Persist Best Model and Scaler
    best_model = models[best_model_name]["model"]
    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    summary_payload = {
        "best_model": best_model_name,
        "evaluation_timestamp": pd.Timestamp.now().isoformat(),
        "total_records": n_samples,
        "train_records": len(train_df),
        "test_records": len(test_df),
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "model_results": results
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(summary_payload, f, indent=2)

    logging.info(f"Selected Best Model: {best_model_name} (Test RMSE: {best_rmse:.4f})")
    logging.info(f"Saved metrics summary to {METRICS_PATH}")
    return summary_payload


if __name__ == "__main__":
    train_and_compare_models()
