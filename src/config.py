"""
Configuration module for XAU/USD Gold Price Prediction ML project.
"""
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
FRONTEND_DIR = BASE_DIR / "frontend"

# Ensure essential directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# File Paths
RAW_DATA_PATH = DATA_DIR / "xauusd_historical.csv"
PROCESSED_DATA_PATH = DATA_DIR / "xauusd_features.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
RF_MODEL_PATH = MODELS_DIR / "rf_model.joblib"
SVR_MODEL_PATH = MODELS_DIR / "svr_model.joblib"
GB_MODEL_PATH = MODELS_DIR / "gb_model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Twelve Data API Settings
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "")
TWELVE_DATA_BASE_URL = "https://api.twelvedata.com/time_series"
SYMBOL = "XAU/USD"
DEFAULT_INTERVAL = "1h"

# Feature Engineering Parameters
FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "sma_20", "ema_50", "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width", "atr_14",
    "return_1", "return_3", "return_5",
    "volatility_10", "volatility_20",
    "hour", "day_of_week", "month", "is_london_ny_overlap"
]

TARGET_COLUMN = "future_close_1"
DIRECTION_COLUMN = "trend_direction"
