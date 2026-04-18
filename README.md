# Stock Price Prediction

**FRE-GY 7773: Machine Learning in Financial Engineering**
**Group 2**: Zhang Xiaoyang, Zhao Zhiqi, Gupta Prachee

---

## Overview

We build a machine learning pipeline to predict next-day stock return direction (up/down) using historical price data and technical indicators. We evaluate models out-of-sample and backtest the corresponding trading strategy.

## Methods

- **Baseline**: Linear Regression
- **Main Model**: Logistic Regression with L1 (Lasso) and L2 (Ridge) regularization
- **Feature Selection**: Lasso penalty to identify the most predictive indicators
- **Dimensionality Reduction**: PCA (for comparison)

## Features

Technical indicators constructed from daily OHLCV data:

| Feature | Description |
|---|---|
| MA5, MA20 | 5-day and 20-day moving averages |
| MA\_ratio | MA5 / MA20 (trend signal) |
| Momentum5, Momentum20 | Past 5-day and 20-day returns |
| RSI | Relative Strength Index (14-day) |
| MACD | Moving Average Convergence Divergence |
| BB\_position | Price position within Bollinger Bands |
| Volume\_change | Daily volume change rate |

## Data

- **Source**: Yahoo Finance via `yfinance`
- **Asset**: SPY (S&P 500 ETF)
- **Period**: 2015–2024
- **Train set**: 2015–2022
- **Test set**: 2023–2024

## Repository Structure

```
stock-price-prediction/
├── data/               # Processed data (raw data downloaded via notebook)
├── notebooks/
│   ├── 01_data_features.ipynb    # Data download & feature engineering
│   ├── 02_modeling.ipynb         # Model training & evaluation
│   └── 03_backtest.ipynb         # Trading strategy & backtest
├── src/
│   ├── features.py     # Feature engineering functions
│   ├── models.py       # Model training & evaluation
│   └── backtest.py     # Backtesting utilities
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

*(To be filled after experiments)*

## Limitations & Future Work

*(To be filled after experiments)*
