"""
FastAPI Backend REST API for XAU/USD Gold Market Price Prediction.
Serves:
- REST API for live/latest prediction and custom feature prediction
- Model evaluation metrics comparison (Random Forest vs SVR vs Gradient Boosting)
- Historical OHLCV + technical indicators data for frontend charting
- Twelve Data API live refresh endpoint
- Static file serving for the frontend dashboard
"""
import sys
import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import pandas as pd
from src.config import (
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    METRICS_PATH,
    FRONTEND_DIR,
    TWELVE_DATA_API_KEY
)
from src.predict import GoldPricePredictor
from src.data_loader import fetch_from_twelve_data, audit_dataset_quality
from src.feature_engineering import apply_feature_engineering

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(
    title="XAU/USD Gold Price Prediction API",
    description="Machine Learning REST Service for XAU/USD Market Forecasting",
    version="1.0.0"
)

# Enable CORS for local cross-origin development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Predictor Instances
try:
    best_predictor = GoldPricePredictor(model_type="best")
    rf_predictor = GoldPricePredictor(model_type="rf")
    svr_predictor = GoldPricePredictor(model_type="svr")
except Exception as e:
    logging.warning(f"Predictors not initialized yet: {e}")
    best_predictor = None
    rf_predictor = None
    svr_predictor = None


class CustomPredictionRequest(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float = 3500.0
    model_choice: Optional[str] = "best"


class FetchLiveDataRequest(BaseModel):
    api_key: Optional[str] = None
    symbol: Optional[str] = "XAU/USD"
    interval: Optional[str] = "1h"
    outputsize: Optional[int] = 500


@app.get("/api/health")
def health_check():
    """Service health and models status."""
    return {
        "status": "healthy",
        "service": "XAU/USD Gold Market Predictor",
        "models_loaded": best_predictor is not None,
        "processed_data_exists": PROCESSED_DATA_PATH.exists()
    }


@app.get("/api/metrics")
def get_model_metrics():
    """Return model comparison metrics (MAE, RMSE, R2, Directional Accuracy, Feature Importance)."""
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Metrics not found. Please train models first.")
    with open(METRICS_PATH, "r") as f:
        metrics_data = json.load(f)
    return metrics_data


@app.get("/api/historical")
def get_historical_data(limit: int = Query(default=150, ge=1, le=1000)):
    """Return recent historical candles and technical indicators for frontend charts."""
    if not PROCESSED_DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Engineered dataset not found.")
    
    df = pd.read_csv(PROCESSED_DATA_PATH)
    recent = df.tail(limit).copy()

    records = []
    for _, row in recent.iterrows():
        records.append({
            "datetime": str(row["datetime"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "sma_20": float(row.get("sma_20", row["close"])),
            "ema_50": float(row.get("ema_50", row["close"])),
            "rsi_14": float(row.get("rsi_14", 50.0)),
            "macd": float(row.get("macd", 0.0)),
            "macd_signal": float(row.get("macd_signal", 0.0)),
            "bb_upper": float(row.get("bb_upper", row["close"])),
            "bb_lower": float(row.get("bb_lower", row["close"]))
        })

    return {
        "count": len(records),
        "symbol": "XAU/USD",
        "interval": "1h",
        "candles": records
    }


@app.get("/api/predict/latest")
def predict_latest(model: str = Query(default="best", enum=["best", "rf", "svr"])):
    """Forecast the next hour's gold closing price using the latest available market state."""
    pred_instance = {
        "best": best_predictor,
        "rf": rf_predictor,
        "svr": svr_predictor
    }.get(model, best_predictor)

    if pred_instance is None:
        raise HTTPException(status_code=503, detail="Prediction models are not loaded.")

    try:
        result = pred_instance.predict_latest_from_dataset()
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/predict")
def predict_custom(req: CustomPredictionRequest):
    """
    Predict next price given manual user-entered OHLC candle values.
    Synthesizes required technical indicators relative to latest baseline.
    """
    pred_instance = {
        "best": best_predictor,
        "rf": rf_predictor,
        "svr": svr_predictor
    }.get(req.model_choice, best_predictor)

    if pred_instance is None:
        raise HTTPException(status_code=503, detail="Prediction models are not loaded.")

    # Load recent data to compute updated technical indicators with the user's candle appended
    df = pd.read_csv(PROCESSED_DATA_PATH)
    latest_row = df.iloc[-1].to_dict()

    # Create dummy new candle
    new_candle = {
        "datetime": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "open": req.open,
        "high": req.high,
        "low": req.low,
        "close": req.close,
        "volume": req.volume
    }

    # Append to recent raw history and recalculate features
    raw_df = pd.read_csv(RAW_DATA_PATH)
    temp_df = pd.concat([raw_df.tail(60), pd.DataFrame([new_candle])], ignore_index=True)
    features_df = apply_feature_engineering(temp_df)
    features_dict = features_df.iloc[-1].to_dict()

    prediction = pred_instance.predict_from_features(features_dict)
    prediction["input_candle"] = new_candle
    return prediction


@app.post("/api/fetch-live")
def fetch_live_market_data(req: FetchLiveDataRequest):
    """
    Trigger live data ingestion from Twelve Data API.
    Updates the historical data file and regenerates features.
    """
    api_key = req.api_key or os.getenv("TWELVE_DATA_API_KEY", TWELVE_DATA_API_KEY)
    if not api_key or api_key == "your_twelve_data_api_key_here":
        raise HTTPException(
            status_code=400,
            detail="Twelve Data API Key missing. Please provide a valid key from https://twelvedata.com/login"
        )

    try:
        df_live = fetch_from_twelve_data(api_key=api_key, symbol=req.symbol, interval=req.interval, outputsize=req.outputsize)
        df_live.to_csv(RAW_DATA_PATH, index=False)

        # Update engineered features
        df_features = apply_feature_engineering(df_live)
        df_features.to_csv(PROCESSED_DATA_PATH, index=False)

        return {
            "status": "success",
            "message": f"Successfully pulled {len(df_live)} live candles from Twelve Data.",
            "audit": audit_dataset_quality(df_live)
        }
    except Exception as e:
        logging.error(f"Error fetching Twelve Data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Serve Frontend Web Application
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    # Additional mount to handle direct relative folder paths if requested
    app.mount("/xauusd-gold-prediction/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend_alias")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        from fastapi.responses import Response
        return Response(status_code=204)

    @app.get("/")
    def serve_index():
        return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
