"""
Feature Engineering Module for XAU/USD Gold Price Prediction.
Implements the 6 mandatory feature engineering techniques specified in the assignment:
1. Technical Indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR)
2. Lagged Momentum Returns (1-step, 3-step, 5-step)
3. Rolling Volatility Aggregations (10-period, 20-period)
4. Date/Time Feature Extraction & Market Session Encoding
5. Outlier Treatment & Winsorization (IQR clipping)
6. Feature Scaling & Leakage-Free Standardization
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
import joblib
from src.config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    DIRECTION_COLUMN,
    SCALER_PATH,
    PROCESSED_DATA_PATH
)


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate MACD, Signal line, and Histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def calculate_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    """Calculate Bollinger Bands and Bandwidth."""
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + (num_std * std)
    lower = sma - (num_std * std)
    width = (upper - lower) / (sma + 1e-10)
    return upper, lower, width


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)."""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr


def apply_feature_engineering(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full feature engineering pipeline to raw OHLCV DataFrame.
    """
    df = df_raw.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # -------------------------------------------------------------
    # Technique 1: Technical Indicators Creation
    # -------------------------------------------------------------
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["rsi_14"] = calculate_rsi(df["close"], period=14)
    df["macd"], df["macd_signal"], df["macd_hist"] = calculate_macd(df["close"])
    df["bb_upper"], df["bb_lower"], df["bb_width"] = calculate_bollinger_bands(df["close"])
    df["atr_14"] = calculate_atr(df, period=14)

    # -------------------------------------------------------------
    # Technique 2: Lag Features & Multi-Period Returns
    # -------------------------------------------------------------
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)

    # -------------------------------------------------------------
    # Technique 3: Rolling Volatility & Statistical Aggregations
    # -------------------------------------------------------------
    df["volatility_10"] = df["return_1"].rolling(window=10).std()
    df["volatility_20"] = df["return_1"].rolling(window=20).std()

    # -------------------------------------------------------------
    # Technique 4: Date/Time Feature Extraction & Session Interaction
    # -------------------------------------------------------------
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    # London / New York market overlap (12:00 to 16:00 UTC) has peak liquidity
    df["is_london_ny_overlap"] = df["hour"].apply(lambda h: 1 if 12 <= h <= 16 else 0)

    # -------------------------------------------------------------
    # Technique 5: Outlier Treatment & Capping (IQR Method)
    # -------------------------------------------------------------
    # Cap extreme anomalies in volume and returns to prevent distortion
    for col in ["volume", "return_1"]:
        q1 = df[col].quantile(0.01)
        q3 = df[col].quantile(0.99)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    # -------------------------------------------------------------
    # Target Construction (Next Period Forecasting)
    # -------------------------------------------------------------
    df[TARGET_COLUMN] = df["close"].shift(-1)
    df["future_return_1"] = (df[TARGET_COLUMN] - df["close"]) / df["close"]
    df[DIRECTION_COLUMN] = (df[TARGET_COLUMN] > df["close"]).astype(int)

    # Drop warm-up NaN rows caused by rolling windows and the final target shift
    df = df.dropna().reset_index(drop=True)
    return df


def prepare_scaled_features(train_df: pd.DataFrame, test_df: pd.DataFrame = None, fit_scaler: bool = True, scaler=None):
    """
    Technique 6: Feature Scaling & Leakage-Free Standardization.
    Scales features using RobustScaler (resilient to financial extremes).
    Scaler is fitted strictly on the training set to prevent lookahead bias.
    """
    if fit_scaler:
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(train_df[FEATURE_COLUMNS])
    else:
        if scaler is None:
            raise ValueError("Scaler instance must be provided when fit_scaler=False")
        X_train_scaled = scaler.transform(train_df[FEATURE_COLUMNS])

    if test_df is not None:
        X_test_scaled = scaler.transform(test_df[FEATURE_COLUMNS])
        return X_train_scaled, X_test_scaled, scaler

    return X_train_scaled, scaler


def run_feature_engineering_pipeline(save_csv: bool = True) -> pd.DataFrame:
    """Load raw data, engineer all 6 feature sets, and persist processed dataset."""
    from src.data_loader import load_or_fetch_dataset

    df_raw = load_or_fetch_dataset()
    df_features = apply_feature_engineering(df_raw)

    if save_csv:
        df_features.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"Engineered dataset saved to {PROCESSED_DATA_PATH} ({len(df_features)} records, {df_features.shape[1]} columns)")

    return df_features


if __name__ == "__main__":
    df = run_feature_engineering_pipeline()
    print("\nFeature Engineering Verification:")
    print("Columns:", list(df.columns))
    print("Shape:", df.shape)
    print("\nSample records:")
    print(df[["datetime", "close", "rsi_14", "macd", "bb_width", TARGET_COLUMN, DIRECTION_COLUMN]].tail())
