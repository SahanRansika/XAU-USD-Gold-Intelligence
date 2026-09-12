# Machine Learning Module - Group Project Assignment Report
## Project Title: XAU/USD (Gold) Market Price Prediction Model & Full-Stack Application

**Module**: Machine Learning & Full-Stack Integration  
**Submission Deadline**: 18 September  
**Target Asset**: XAU/USD (Gold vs. US Dollar Spot Market)  
**Primary Data Provider**: [Twelve Data API](https://twelvedata.com/login)  

---

## 1. Assignment Overview & Problem Statement

### 1.1 Problem Selection
Financial markets, particularly commodity markets such as Gold (XAU/USD), are notoriously non-linear, non-stationary, and prone to sudden volatility regimes influenced by macroeconomic factors, interest rate announcements, and geopolitical risks. Traditional quantitative approaches often struggle to adapt to changing market microstructures. 

The objective of this project is to develop an end-to-end machine learning solution to forecast the **future closing price ($Close_{t+1}$)** and the **expected directional movement (Bullish / Bearish)** of XAU/USD on an hourly resolution.

### 1.2 Real-World Application
- **Algorithmic Trading & Execution**: Providing actionable predictive signals to automated trading strategies.
- **Risk & Portfolio Hedging**: Assisting institutional and retail traders in forecasting near-term volatility and price drawdown risks.
- **Decision Support System**: Supplying interactive visual overlays and scenario simulation tools for financial analysts.

---

## 2. Dataset Description & Data Quality Audit

### 2.1 Dataset Source
Market data is acquired through the **[Twelve Data API](https://twelvedata.com/)** (`https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h`), a financial data provider delivering real-time and historical forex and commodity prices. In addition, an offline benchmark historical dataset is packaged with the project to ensure reproducible training and offline accessibility.

### 2.2 Dataset Attributes
| Metric | Specification |
| :--- | :--- |
| **Asset Symbol** | `XAU/USD` (Troy ounce of Gold in US Dollars) |
| **Time Horizon** | Hourly candles (`1h` interval) |
| **Total Records** | 3,000 raw candles (2,979 post-feature warm-up) |
| **Raw Input Features** | 6 columns (`datetime`, `open`, `high`, `low`, `close`, `volume`) |
| **Target Variables** | `future_close_1` (Continuous), `trend_direction` (Binary: 1 = Up, 0 = Down) |

### 2.3 Data Quality Audit
A data quality inspection was conducted via `src/data_loader.py` with the following verification:
1. **Missing Values**: $0$ missing records detected across all OHLCV columns.
2. **Duplicate Records**: $0$ duplicate timestamps; time-series monotonicity was verified.
3. **Price Sanity**: Lowest observed close: $\$2,154.53$, Highest close: $\$2,888.88$, Mean close: $\$2,559.43$.
4. **Data Types**: `datetime` parsed as ISO datetime64; all price features cast to 64-bit floating-point numbers; volume cast to 64-bit integers.

---

## 3. Mandatory Feature Engineering (6 Techniques Applied)

In compliance with assignment requirements mandating **at least 5–6 meaningful feature engineering techniques**, the following domain-relevant transformations were implemented in `src/feature_engineering.py`:

### Technique 1: Technical Indicators Creation
Calculated mathematical indicators used by technical analysts to capture trend, momentum, and volatility:
- **Simple Moving Average (SMA-20)**: $SMA_{20} = \frac{1}{20}\sum_{i=0}^{19} Close_{t-i}$ (trend baseline).
- **Exponential Moving Average (EMA-50)**: Gives higher weighting to recent prices to detect medium-term trend shifts.
- **Relative Strength Index (RSI-14)**: Measures the velocity and magnitude of price movements on an oscillator scale of $0$ to $100$.
- **Moving Average Convergence Divergence (MACD)**: Fast EMA (12) minus Slow EMA (26), paired with a 9-period signal line and histogram.
- **Bollinger Bands (Upper, Lower, Width)**: Measures volatility expansion and compression (2 standard deviations around the 20-period SMA).
- **Average True Range (ATR-14)**: Quantifies true market volatility across high/low gaps.

### Technique 2: Multi-Step Lagged Momentum Returns
Raw price levels are non-stationary. To make the series stationary and extract momentum signals, percentage price changes over multiple lookback windows were computed:
$$\text{Return}_k = \frac{Close_t - Close_{t-k}}{Close_{t-k}} \quad \text{for } k \in \{1, 3, 5\}$$
These features capture short-term price velocity and mean-reversion tendencies.

### Technique 3: Rolling Statistical Volatility Aggregation
Financial time series exhibit volatility clustering. We engineered rolling standard deviation features of the 1-step returns:
- `volatility_10`: 10-period rolling standard deviation of returns.
- `volatility_20`: 20-period rolling standard deviation of returns.
These features inform the model whether the market is currently in a quiet consolidation or high-turbulent breakout state.

### Technique 4: Date/Time Feature Extraction & Market Session Encoding
Gold prices exhibit distinct intraday seasonality depending on which global market centers are open:
- Extracted: `hour` (0–23), `day_of_week` (0–6), and `month` (1–12).
- **Session Interaction (`is_london_ny_overlap`)**: A binary indicator representing the 12:00 to 16:00 UTC window where the London and New York trading sessions overlap. This is empirically known as the period of highest liquidity and trading volume for gold.

### Technique 5: Outlier Treatment via IQR Winsorization
Extreme slippage spikes and flash crashes can distort gradient updates and tree split criteria. Using the Interquartile Range (IQR) method:
$$\text{Lower} = Q_1 - 1.5 \times IQR, \quad \text{Upper} = Q_3 + 1.5 \times IQR$$
Extreme anomalies in `volume` and `return_1` were clipped at the 1st and 99th percentile boundaries, preserving record count while mitigating outlier distortion.

### Technique 6: Leakage-Free Feature Scaling
Features such as price ($\sim \$2,700$), volume ($\sim 4,000$), RSI ($0-100$), and percentage returns ($\pm 0.02$) have vastly different numerical scales.
- Used **`RobustScaler`**, which scales features using the median and interquartile range, rendering it resilient to remaining distribution tails.
- **Lookahead Bias Prevention**: The scaler was fitted **strictly on the training partition** ($80\%$) and only used to transform the test partition ($20\%$).

---

## 4. Machine Learning Model Development & Evaluation

### 4.1 Candidate Algorithms
In accordance with the project proposal, traditional machine learning models were developed and compared:
1. **Random Forest Regressor**: Non-linear ensemble model combining 100 decorrelated decision trees.
2. **Support Vector Regression (SVR)**: Kernel-based model using the Radial Basis Function (RBF) kernel with hyperparameter tuning ($C=100.0, \epsilon=0.1$).
3. **Gradient Boosting Regressor**: Sequential ensemble minimizing squared error loss.

### 4.2 Cross-Validation Strategy
Standard random k-fold cross-validation is invalid for financial time-series because shuffling future data into the training set introduces fatal lookahead bias. We utilized **`TimeSeriesSplit` (5 folds)**, ensuring each training split strictly precedes the validation fold in time.

### 4.3 Model Performance Comparison

| Model | 5-Fold CV RMSE | Test MAE | Test RMSE | $R^2$ Score | Directional Accuracy | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest Regressor** | **$66.90** | **$8.33** | **$13.77** | **91.58%** | **51.85%** | **Selected Best Model** |
| **Gradient Boosting Regressor** | $68.96 | $8.82 | $14.87 | 90.18% | 51.34% | Benchmark |
| **Support Vector Regression (SVR)** | $80.02 | $30.02 | $43.56 | 15.71% | 48.66% | Baseline |

### 4.4 Findings & Feature Importance
- **Random Forest Regressor** achieved the lowest prediction error ($RMSE = \$13.77$, $MAE = \$8.33$) and explained **$91.58\%$** of price variance on unseen test data.
- Feature importance analysis revealed that the most influential predictors were:
  1. `close` & `sma_20` (Current price anchoring)
  2. `bb_upper` & `bb_lower` (Dynamic volatility envelopes)
  3. `ema_50` (Intermediate trend momentum)
  4. `atr_14` & `volatility_20` (Market risk regime)

---

## 5. Full-Stack Application Architecture

The application implements the complete 6-tier architecture specified in the assignment brief:

```
[User]
   │
   ▼
[Frontend Application] (Single-Page Trading Terminal: HTML5, CSS3, Chart.js)
   │ (HTTP REST / JSON)
   ▼
[Backend REST API] (FastAPI Asynchronous Engine on port 8000)
   │
   ▼
[ML Prediction Service] (GoldPricePredictor in src/predict.py)
   │
   ▼
[Trained ML Model & Scaler] (best_model.joblib & scaler.joblib)
   │
   ▼
[Prediction Result: Price Forecast + Bullish/Bearish Direction + Confidence]
   │
   ▼
[Rendered on Interactive Charts & Real-Time Signal Badges in Frontend]
```

### Full-Stack Features:
1. **Interactive Charting**: Displays Gold prices alongside SMA-20, EMA-50, and Bollinger Bands with interactive visibility toggles.
2. **Sub-Chart Indicator Switcher**: Allows toggling between RSI (14) with overbought/oversold bands and MACD oscillator histograms.
3. **Real-Time Prediction Hero**: Generates next-period target price, expected delta, and directional classification badges.
4. **Custom Scenario Simulator**: Allows users to enter hypothetical OHLCV candle parameters to test the model's predictive response.
5. **Twelve Data API Ingestion Modal**: Enables users to configure their Twelve Data API key to pull live market feeds directly into the pipeline.

---

## 6. Group Collaboration & Viva Voce Preparation Guide

For groups of up to 4 students, responsibilities can be divided as follows:

| Role / Student | Responsibilities | Key Viva Voce Talking Points |
| :--- | :--- | :--- |
| **Student 1: Data Engineer** | Twelve Data API integration, data ingestion pipeline (`src/data_loader.py`), data cleaning, quality auditing. | Explain how API pagination/limits work, why duplicate timestamp checks matter, how missing values in financial series are handled. |
| **Student 2: ML & Feature Engineer** | Design and implementation of the 6 feature engineering techniques (`src/feature_engineering.py`). | Explain the financial meaning of RSI and MACD, how IQR clipping prevents flash crash distortion, and why `RobustScaler` was chosen. |
| **Student 3: Model Developer** | TimeSeries cross-validation, model training (RF, SVR, GB), hyperparameter tuning, model evaluation (`src/train.py`, `src/evaluate.py`). | Explain why standard k-fold CV cannot be used for time series, how Directional Accuracy is calculated, and why Random Forest outperformed SVR. |
| **Student 4: Full-Stack Engineer** | FastAPI REST API development (`backend/app.py`), UI design, interactive Chart.js frontend (`frontend/`), system integration. | Explain REST API design, endpoint response formats, how the frontend dynamically communicates with the ML inference service. |

### Anticipated Viva Voce Questions & Answers

1. **Q: Why can't you use standard train_test_split (random split) for time-series forecasting?**  
   *A*: Financial data has temporal dependencies and autocorrelation. A random split causes data leakage by using future prices to predict past prices, yielding unrealistically optimistic metrics that fail in live trading. We used chronological splitting and `TimeSeriesSplit` cross-validation to maintain causality.

2. **Q: Why did Random Forest outperform SVR in this experiment?**  
   *A*: Financial features have non-linear threshold effects (e.g., RSI crossing 70 or 30). Decision trees naturally create orthogonal decision boundaries that capture these threshold rules, whereas SVR requires careful hyperparameter tuning of $C$, $\gamma$, and $\epsilon$ and can be sensitive to market regime shifts.

3. **Q: What is Directional Accuracy and why is it important alongside RMSE?**  
   *A*: In financial trading, a model could have a low RMSE by simply predicting a price very close to the current price, but it might get the direction (Up or Down) wrong. Directional Accuracy measures what percentage of time the model correctly anticipated whether the market would rise or fall, which dictates trade profitability.
