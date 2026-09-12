# XAU/USD (Gold) Market Price Prediction Model & Full-Stack Application

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![Twelve Data](https://img.shields.io/badge/Data-Twelve%20Data%20API-gold.svg)](https://twelvedata.com/)

An end-to-end Machine Learning pipeline and full-stack interactive trading dashboard designed to forecast price movements and market directions of **XAU/USD (Gold vs. US Dollar)**.

Developed in compliance with the **Machine Learning Module - Group Project Assignment** requirements.

---

## 🌟 Key Features

1. **Dual Data Ingestion Pipeline**:
   - Live streaming and historical candle downloads via the [Twelve Data API](https://twelvedata.com/login).
   - High-fidelity offline historical dataset with realistic gold market dynamics.
2. **6 Mandatory Feature Engineering Techniques**:
   - Domain-specific Technical Indicators: SMA (20), EMA (50), RSI (14), MACD (12, 26, 9), Bollinger Bands, ATR.
   - Lagged Momentum Returns (1-step, 3-step, 5-step).
   - Rolling Statistical Volatility (10-period & 20-period standard deviation).
   - Temporal & Session Extraction: Hour, Day of Week, Month, London/NY Session Overlap.
   - Robust Outlier Capping (Interquartile Range - IQR clipping).
   - Leakage-Free Feature Scaling (`RobustScaler` fitted strictly on training partitions).
3. **Multi-Model Machine Learning Development**:
   - **Random Forest Regressor** (Best Model: $R^2 = 91.58\%$, $RMSE = \$13.77$, Directional Accuracy = $51.85\%$).
   - **Support Vector Regression (SVR)** with RBF Kernel.
   - **Gradient Boosting Regressor** Benchmark.
   - 5-Fold `TimeSeriesSplit` cross-validation preventing temporal lookahead bias.
4. **Full-Stack Application**:
   - **FastAPI REST API**: High-speed asynchronous endpoints for inference, metrics, and data refresh.
   - **Interactive Trading Dashboard**: Real-time Chart.js charting, technical indicator overlays, scenario simulator, and model evaluation scoreboard.

---

## 📁 Project Architecture

```
xauusd-gold-prediction/
├── .env.example                  # Template for Twelve Data API Key and config
├── requirements.txt              # Project dependencies
├── README.md                     # Setup and usage guide
├── ASSIGNMENT_REPORT.md          # Comprehensive academic assignment report
│
├── data/
│   ├── xauusd_historical.csv    # Raw historical OHLCV data
│   └── xauusd_features.csv      # Engineered feature dataset (28 columns)
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # Hyperparameters, column definitions, paths
│   ├── data_loader.py            # Twelve Data API ingestion & data quality auditing
│   ├── feature_engineering.py    # The 6 mandatory feature engineering transformations
│   ├── train.py                  # Model training, TimeSeriesSplit CV, metric evaluation
│   ├── evaluate.py               # Visual plots and residual error analysis
│   └── predict.py                # Standalone inference service
│
├── models/
│   ├── best_model.joblib         # Serialized best model (Random Forest)
│   ├── rf_model.joblib           # Serialized Random Forest model
│   ├── svr_model.joblib          # Serialized SVR model
│   ├── gb_model.joblib           # Serialized Gradient Boosting model
│   ├── scaler.joblib             # Serialized fitted RobustScaler
│   └── metrics.json              # Model evaluation metrics and benchmarks
│
├── reports/
│   ├── forecast_comparison.png   # Actual vs Predicted price trajectory plot
│   ├── feature_importance.png    # Top Gini feature importance bar chart
│   └── residual_distribution.png # Residual prediction error histogram
│
├── backend/
│   ├── __init__.py
│   └── app.py                    # FastAPI REST application
│
└── frontend/
    ├── index.html                # Dark-mode trading terminal UI
    ├── style.css                 # Professional financial stylesheet
    └── app.js                    # Chart.js controller and API integration
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+ installed
- Free Twelve Data API Key from [Twelve Data](https://twelvedata.com/login) (optional, offline fallback included)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Configure Twelve Data API
Copy the `.env.example` file to `.env` and insert your API key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
TWELVE_DATA_API_KEY=your_twelve_data_api_key_here
```

### 4. Run the ML Pipeline (Data, Features, Model Training)
```bash
# 1. Inspect data and run data quality audit
python src/data_loader.py

# 2. Run feature engineering (implements all 6 techniques)
python src/feature_engineering.py

# 3. Train models and evaluate cross-validation metrics
python src/train.py

# 4. Generate evaluation plots into reports/
python src/evaluate.py
```

### 5. Launch the Full-Stack Web Application
```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Open your web browser and navigate to:
```
http://127.0.0.1:8000/
```

Interactive Swagger REST API Documentation is available at:
```
http://127.0.0.1:8000/docs
```

---

## 📊 Model Performance Summary

| Model Name | 5-Fold CV RMSE | Test MAE | Test RMSE | $R^2$ Score | Directional Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest Regressor** ⭐ | **$66.90** | **$8.33** | **$13.77** | **91.58%** | **51.85%** |
| **Gradient Boosting Regressor** | $68.96 | $8.82 | $14.87 | 90.18% | 51.34% |
| **Support Vector Regression (SVR)** | $80.02 | $30.02 | $43.56 | 15.71% | 48.66% |

---

## 🔌 API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the interactive trading dashboard UI |
| `GET` | `/api/health` | Service health status and model load verification |
| `GET` | `/api/metrics` | Returns evaluation metrics comparison and feature importances |
| `GET` | `/api/historical?limit=100` | Returns recent OHLCV and indicator data for charting |
| `GET` | `/api/predict/latest?model=best` | Predicts next close price for current market state |
| `POST` | `/api/predict` | Runs inference for custom user-submitted OHLC candle |
| `POST` | `/api/fetch-live` | Pulls live market data from Twelve Data API |

---

## 👥 Authors & Academic Context
Developed as part of the **Machine Learning Module - Group Project Assignment**. For full methodology, feature engineering explanations, and viva voce preparation, consult [ASSIGNMENT_REPORT.md](ASSIGNMENT_REPORT.md).
