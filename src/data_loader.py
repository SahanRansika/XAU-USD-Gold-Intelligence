"""
Data acquisition, Twelve Data API ingestion, and historical dataset loader.
"""
import os
import sys
import json
import logging
from pathlib import Path

# Add project root to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.config import (
    RAW_DATA_PATH,
    TWELVE_DATA_API_KEY,
    TWELVE_DATA_BASE_URL,
    SYMBOL,
    DEFAULT_INTERVAL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def fetch_from_twelve_data(api_key: str = None, symbol: str = SYMBOL, interval: str = DEFAULT_INTERVAL, outputsize: int = 2000) -> pd.DataFrame:
    """
    Fetch OHLCV time-series data from Twelve Data API.
    URL: https://twelvedata.com/
    """
    key = api_key or os.getenv("TWELVE_DATA_API_KEY", TWELVE_DATA_API_KEY)
    if not key or key == "your_twelve_data_api_key_here":
        raise ValueError("Twelve Data API key is not configured. Please supply a valid API key.")

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": key,
        "format": "JSON"
    }

    logging.info(f"Fetching {symbol} ({interval}) data from Twelve Data API...")
    response = requests.get(TWELVE_DATA_BASE_URL, params=params, timeout=15)
    data = response.json()

    if "values" not in data:
        error_msg = data.get("message", "Unknown error from Twelve Data API")
        raise RuntimeError(f"Twelve Data API error: {error_msg}")

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    numeric_cols = ["open", "high", "low", "close"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    else:
        df["volume"] = 1000

    df = df.sort_values("datetime").reset_index(drop=True)
    logging.info(f"Successfully fetched {len(df)} candles from Twelve Data.")
    return df


def generate_realistic_xauusd_data(n_candles: int = 3000, start_price: float = 2320.0) -> pd.DataFrame:
    """
    Synthesize realistic XAU/USD gold hourly candles based on geometric Brownian motion
    with mean-reverting drift, realistic volatility, and intraday volume profiles.
    Used for instant offline development and baseline training.
    """
    logging.info(f"Generating realistic synthetic historical XAU/USD dataset ({n_candles} candles)...")
    np.random.seed(42)

    end_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    timestamps = [end_time - timedelta(hours=i) for i in range(n_candles)][::-1]

    # Geometric Brownian motion with intraday seasonality
    dt = 1.0 / (24 * 252)
    mu = 0.08  # ~8% annual trend for gold
    sigma = 0.16  # ~16% annualized volatility
    
    prices = [start_price]
    for i in range(1, n_candles):
        hour = timestamps[i].hour
        # Higher volatility during London/NY trading sessions (8:00 - 18:00 UTC)
        vol_multiplier = 1.6 if 8 <= hour <= 18 else 0.7
        daily_drift = (mu - 0.5 * (sigma * vol_multiplier) ** 2) * dt
        shock = (sigma * vol_multiplier) * np.sqrt(dt) * np.random.normal()
        new_price = prices[-1] * np.exp(daily_drift + shock)
        prices.append(new_price)

    prices = np.array(prices)

    # Construct OHLC with realistic intra-candle wicks and spreads
    candle_noise = np.random.uniform(0.001, 0.0035, size=n_candles)
    high_mult = 1.0 + np.abs(np.random.normal(0, candle_noise))
    low_mult = 1.0 - np.abs(np.random.normal(0, candle_noise))

    highs = np.maximum(prices * high_mult, prices)
    lows = np.minimum(prices * low_mult, prices)
    opens = [prices[0]] + list(prices[:-1])
    closes = prices

    for i in range(n_candles):
        highs[i] = max(highs[i], opens[i], closes[i]) + np.random.uniform(0.1, 0.5)
        lows[i] = min(lows[i], opens[i], closes[i]) - np.random.uniform(0.1, 0.5)

    base_volumes = np.random.randint(1500, 6500, size=n_candles)
    # Session volume boost
    session_boost = np.array([2.2 if 8 <= ts.hour <= 17 else 1.0 for ts in timestamps])
    volumes = (base_volumes * session_boost).astype(int)

    df = pd.DataFrame({
        "datetime": timestamps,
        "open": np.round(opens, 2),
        "high": np.round(highs, 2),
        "low": np.round(lows, 2),
        "close": np.round(closes, 2),
        "volume": volumes
    })

    return df


def audit_dataset_quality(df: pd.DataFrame) -> dict:
    """
    Perform a formal data quality audit as required by the assignment brief:
    - Number of records & features
    - Missing value analysis
    - Duplicate detection
    - Data types & sanity checks
    """
    missing_counts = df.isnull().sum().to_dict()
    duplicates = int(df.duplicated(subset=["datetime"]).sum())
    
    report = {
        "record_count": int(len(df)),
        "feature_count": int(df.shape[1]),
        "features": list(df.columns),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {k: int(v) for k, v in missing_counts.items()},
        "duplicate_timestamps": duplicates,
        "date_range": {
            "start": str(df["datetime"].min()),
            "end": str(df["datetime"].max())
        },
        "price_summary": {
            "min_close": float(df["close"].min()),
            "max_close": float(df["close"].max()),
            "mean_close": float(df["close"].mean()),
            "latest_close": float(df["close"].iloc[-1])
        }
    }
    return report


def load_or_fetch_dataset(force_api: bool = False, api_key: str = None) -> pd.DataFrame:
    """
    Loads dataset: attempts Twelve Data API if key provided; otherwise loads/generates
    local historical dataset.
    """
    if force_api or (api_key and api_key != "your_twelve_data_api_key_here"):
        try:
            df = fetch_from_twelve_data(api_key=api_key)
            df.to_csv(RAW_DATA_PATH, index=False)
            logging.info(f"Saved live Twelve Data to {RAW_DATA_PATH}")
            return df
        except Exception as exc:
            logging.warning(f"Failed to fetch live data from Twelve Data: {exc}. Falling back to local/cached data.")

    if RAW_DATA_PATH.exists():
        logging.info(f"Loading cached historical data from {RAW_DATA_PATH}")
        df = pd.read_csv(RAW_DATA_PATH)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    # Generate initial realistic dataset and save
    df = generate_realistic_xauusd_data()
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    logging.info(f"Cached generated dataset to {RAW_DATA_PATH}")
    return df


if __name__ == "__main__":
    df = load_or_fetch_dataset()
    audit = audit_dataset_quality(df)
    print("\n" + "="*50)
    print("DATASET QUALITY AUDIT REPORT")
    print("="*50)
    print(json.dumps(audit, indent=2))
