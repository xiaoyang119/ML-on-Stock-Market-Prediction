# Stock Price Prediction

**FRE-GY 7773: Machine Learning in Financial Engineering**
**Group 2**: Zhang Xiaoyang, Zhao Zhiqi, Gupta Prachee

---

## Overview

We build a machine learning pipeline to predict next-day stock return direction (up/down) using historical price data and technical indicators. We evaluate models out-of-sample and backtest the corresponding trading strategy.

## Methods

- **Baseline**: Linear Regression
- **Main Models**: Logistic Regression with L1 (Lasso) and L2 (Ridge) regularization
- **Alternative**: PCA + Logistic Regression, XGBoost
- **Feature Selection**: Lasso penalty to identify the most predictive indicators
- **Dimensionality Reduction**: PCA (for comparison)

## Features

Technical indicators constructed from daily OHLCV data (10 features used in modeling):

| Feature | Description |
|---|---|
| MA\_ratio | MA5 / MA20 (trend signal) |
| Momentum5, Momentum20 | Past 5-day and 20-day returns |
| RSI14 | Relative Strength Index (14-day) |
| MACD, MACD\_signal | Moving Average Convergence Divergence and its signal line |
| BB\_position | Price position within Bollinger Bands |
| Volume\_zscore | Z-score normalized daily volume |
| Overnight\_gap | Overnight gap return |
| VIX | CBOE Volatility Index |

## Data

- **Source**: Yahoo Finance via `yfinance`
- **Asset**: SPY (S&P 500 ETF)
- **Period**: 2015–2024
- **Train set**: 2015–2022
- **Test set**: 2023–2024 (500 trading days)

## Repository Structure

```
stock-price-prediction/
├── data/               # Processed data (raw data downloaded via notebook)
├── notebooks/
│   ├── 01_data_features.ipynb    # Data download & feature engineering
│   ├── 02_modeling.ipynb         # Model training & evaluation
│   └── 03_backtest.ipynb         # Trading strategy & backtest
├── results/            # Figures and metrics
├── pyproject.toml      # uv project config
└── README.md
```

## Installation & Setup

This project uses [uv](https://docs.astral.sh/uv/) for environment management.

```bash
# Clone the repo
git clone https://github.com/<your-username>/stock-price-prediction.git
cd stock-price-prediction

# Create environment and install dependencies
uv sync

# Launch Jupyter
uv run jupyter lab
```

## Results

### Out-of-Sample Classification Accuracy (Test Set: 2023–2024, n=500)

| Model | Accuracy |
|---|---|
| Baseline (Linear Regression) | 48.2% |
| Logistic + Ridge | **57.6%** |
| Logistic + Lasso | 57.0% |
| PCA + Logistic | 56.4% |
| XGBoost | 57.2% |

Logistic Regression with Ridge regularization achieves the best out-of-sample accuracy at **57.6%**, meaningfully above the 50% random baseline. The linear baseline underperforms at 48.2%, confirming that regularization is essential for this task.

### Walk-Forward Validation Accuracy (by Year)

| Test Year | Logistic + Ridge | Logistic + Lasso | XGBoost |
|---|---|---|---|
| 2019 | **59.5%** | 56.0% | 59.1% |
| 2020 | 57.3% | 57.3% | 57.3% |
| 2021 | 58.3% | 58.3% | 58.3% |
| 2022 | 42.6% | 44.2% | 43.0% |
| 2023 | **56.0%** | 55.2% | 55.6% |

Model performance degrades significantly in 2022 (the high-volatility, bear market year), suggesting that directional signals become harder to extract during regime shifts.

### Backtest Performance (Test Set: 2023–2024)

| Strategy | Cumulative Return | Ann. Return | Ann. Vol | Sharpe Ratio | Max Drawdown |
|---|---|---|---|---|---|
| Buy & Hold | 58.82% | 26.26% | 12.83% | 2.047 | -9.97% |
| Baseline (Linear Reg) | -4.95% | -2.52% | 9.62% | -0.262 | -15.51% |
| **Logistic + Ridge** | **58.66%** | **26.19%** | **12.82%** | **2.042** | **-9.97%** |
| Logistic + Lasso | 56.16% | 25.19% | 12.73% | 1.978 | -9.97% |
| PCA + Logistic | 51.79% | 23.41% | 12.62% | 1.855 | -9.97% |
| XGBoost | 56.98% | 25.52% | 12.83% | 1.989 | -9.97% |

The Logistic + Ridge strategy nearly matches Buy & Hold (Sharpe 2.042 vs. 2.047) while the linear baseline destroys value (Sharpe -0.262). This suggests the regularized logistic model successfully filters out noise and captures genuine directional signal.

## Limitations & Future Work

**Limitations:**

- The model is trained on a single asset (SPY), limiting generalizability to other stocks or asset classes.
- Technical indicators are constructed from the same price series used as the prediction target, which may introduce look-ahead bias if not carefully handled at each rebalancing step.
- The 2022 results show that the model struggles during regime changes (high volatility, trend reversals), a known weakness of models trained on historical patterns.
- Transaction costs and slippage are not modeled in the backtest, which would reduce realized returns in practice.
- The prediction horizon is fixed at one day; performance may differ at other horizons.

**Future Work:**

- Incorporate alternative data sources (e.g., sentiment from news/social media, options-implied volatility surface).
- Explore deep learning models (LSTM, Transformer) that can capture longer temporal dependencies.
- Extend to a multi-asset portfolio and incorporate position sizing (e.g., Kelly criterion).
- Add realistic transaction cost modeling and slippage assumptions to the backtest.
- Test robustness across different market regimes using hidden Markov models or regime-switching filters.
