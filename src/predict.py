"""
Inference service for XAU/USD Gold Price Prediction.
Loads saved models & scalers, performs feature preprocessing, and delivers predictions.
"""
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Union

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib

from src.config import (
    BEST_MODEL_PATH,
    RF_MODEL_PATH,
    SVR_MODEL_PATH,
    GB_MODEL_PATH,
    SCALER_PATH,
    METRICS_PATH,
    FEATURE_COLUMNS,
    PROCESSED_DATA_PATH
)
from src.feature_engineering import apply_feature_engineering

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class GoldPricePredictor:
    """Production predictor instance for XAU/USD gold price forecasting."""

    def __init__(self, model_type: str = "best"):
        self.model_type = model_type
        self.scaler = None
        self.model = None
        self.metrics = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Load fitted scaler, metrics, and chosen model."""
        if not SCALER_PATH.exists():
            raise FileNotFoundError(f"Scaler artifact missing at {SCALER_PATH}. Run training first.")
        self.scaler = joblib.load(SCALER_PATH)

        if METRICS_PATH.exists():
            with open(METRICS_PATH, "r") as f:
                self.metrics = json.load(f)

        model_map = {
            "best": BEST_MODEL_PATH,
            "rf": RF_MODEL_PATH,
            "svr": SVR_MODEL_PATH,
            "gb": GB_MODEL_PATH
        }
        target_path = model_map.get(self.model_type.lower(), BEST_MODEL_PATH)
        if not target_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {target_path}")

        self.model = joblib.load(target_path)
        logging.info(f"Loaded predictor using model from: {target_path.name}")

    def predict_from_features(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        """Predict next candle close given pre-calculated feature values."""
        row_values = []
        for col in FEATURE_COLUMNS:
            if col not in feature_dict:
                raise ValueError(f"Missing required feature: '{col}'")
            row_values.append(feature_dict[col])

        features_df = pd.DataFrame([row_values], columns=FEATURE_COLUMNS)
        scaled_features = self.scaler.transform(features_df)
        pred_price = float(self.model.predict(scaled_features)[0])
        current_close = float(feature_dict["close"])
        price_change = pred_price - current_close
        pct_change = (price_change / current_close) * 100.0
        direction = "BULLISH" if price_change > 0 else "BEARISH"

        return {
            "current_price": round(current_close, 2),
            "predicted_price": round(pred_price, 2),
            "predicted_change": round(price_change, 2),
            "predicted_pct_change": round(pct_change, 3),
            "signal": direction,
            "model_used": self.model_type
        }

    def predict_latest_from_dataset(self) -> Dict[str, Any]:
        """Convenience method to predict based on the latest available market candle."""
        if not PROCESSED_DATA_PATH.exists():
            raise FileNotFoundError("Processed dataset not found. Run feature engineering first.")

        df = pd.read_csv(PROCESSED_DATA_PATH)
        latest_row = df.iloc[-1].to_dict()
        pred = self.predict_from_features(latest_row)
        pred["timestamp"] = str(latest_row.get("datetime", "latest"))
        pred["rsi_14"] = round(float(latest_row.get("rsi_14", 50)), 2)
        pred["macd"] = round(float(latest_row.get("macd", 0)), 3)
        pred["sma_20"] = round(float(latest_row.get("sma_20", latest_row["close"])), 2)
        pred["ema_50"] = round(float(latest_row.get("ema_50", latest_row["close"])), 2)
        return pred


if __name__ == "__main__":
    predictor = GoldPricePredictor(model_type="best")
    result = predictor.predict_latest_from_dataset()
    print("\nInference Output:")
    print(json.dumps(result, indent=2))
