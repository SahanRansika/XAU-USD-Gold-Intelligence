"""
Evaluation and visualization script.
Generates:
1. Actual vs Predicted price trajectory plots
2. Feature importance bar chart
3. Residual error distribution
4. Comparative performance table
"""
import sys
import json
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from src.config import (
    PROCESSED_DATA_PATH,
    RF_MODEL_PATH,
    SVR_MODEL_PATH,
    SCALER_PATH,
    METRICS_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    BASE_DIR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def generate_evaluation_visualizations():
    """Produce evaluation charts and save images into reports/."""
    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DATA_PATH)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()

    scaler = joblib.load(SCALER_PATH)
    rf_model = joblib.load(RF_MODEL_PATH)
    svr_model = joblib.load(SVR_MODEL_PATH)

    X_test = scaler.transform(test_df[FEATURE_COLUMNS])
    y_test = test_df[TARGET_COLUMN].values

    rf_preds = rf_model.predict(X_test)
    svr_preds = svr_model.predict(X_test)

    # 1. Price Forecast Comparison Plot (Last 120 test candles)
    sample_len = min(120, len(y_test))
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[-sample_len:], label="Actual XAU/USD Close", color="#f39c12", linewidth=2.2)
    plt.plot(rf_preds[-sample_len:], label="Random Forest Forecast", color="#2ecc71", linestyle="--", linewidth=1.8)
    plt.plot(svr_preds[-sample_len:], label="SVR Forecast", color="#3498db", linestyle=":", linewidth=1.5)
    plt.title("XAU/USD Gold Price Forecast Comparison (Test Holdout)", fontsize=14, fontweight="bold")
    plt.xlabel("Trading Period (Hours)", fontsize=11)
    plt.ylabel("Gold Price (USD / Troy Ounce)", fontsize=11)
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_path = reports_dir / "forecast_comparison.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    logging.info(f"Saved forecast plot to {plot_path}")

    # 2. Random Forest Feature Importance Plot
    if hasattr(rf_model, "feature_importances_"):
        importances = rf_model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1][:12]  # top 12
        top_features = [FEATURE_COLUMNS[i] for i in sorted_indices]
        top_scores = importances[sorted_indices]

        plt.figure(figsize=(10, 6))
        plt.barh(top_features[::-1], top_scores[::-1], color="#16a085")
        plt.title("Top 12 Features by Gini Importance (Random Forest)", fontsize=14, fontweight="bold")
        plt.xlabel("Relative Importance Score", fontsize=11)
        plt.tight_layout()
        importance_path = reports_dir / "feature_importance.png"
        plt.savefig(importance_path, dpi=200)
        plt.close()
        logging.info(f"Saved feature importance plot to {importance_path}")

    # 3. Residual Error Distribution
    residuals = y_test - rf_preds
    plt.figure(figsize=(9, 5))
    plt.hist(residuals, bins=40, color="#9b59b6", edgecolor="black", alpha=0.75)
    plt.title("Random Forest Residual Error Distribution (Actual - Predicted)", fontsize=13, fontweight="bold")
    plt.xlabel("Prediction Error ($ USD)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.axvline(0, color="red", linestyle="dashed", linewidth=1.5)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    residuals_path = reports_dir / "residual_distribution.png"
    plt.savefig(residuals_path, dpi=200)
    plt.close()
    logging.info(f"Saved residual plot to {residuals_path}")


if __name__ == "__main__":
    generate_evaluation_visualizations()
