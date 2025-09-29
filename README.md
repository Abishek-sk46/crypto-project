# 📈 Crypto Price Predictor

## Project Overview
This project analyzes cryptocurrency trends, visualizes patterns, and predicts future prices using a **Random Forest model**. The dashboard is interactive, allowing users to explore historical data, see predictions, and download results.  

**Coins included:**  
- Bitcoin (BTC)  
- Ethereum (ETH)

---

## Features
- **Data Cleaning & Feature Engineering**
  - Moving averages (MA_7, MA_30)  
  - Volatility  
  - Lag features for prediction  
- **Exploratory Data Analysis (EDA)**
  - Historical price charts  
  - Correlation heatmaps  
  - Anomalies detection (>10% daily moves)  
- **Predictive Modeling**
  - Random Forest Regressor for BTC & ETH  
  - Model evaluation metrics: RMSE, MAE, MAPE  
- **Streamlit Dashboard**
  - Coin selector (BTC / ETH)  
  - Interactive historical price chart  
  - Actual vs predicted price chart  
  - Latest price prediction card  
  - Model metrics table  
  - Download predictions as CSV  

---

## Dataset
- **Source:** Historical daily price data from Yahoo Finance / CoinGecko  
- **Time period:** Last 3+ years  
- **Processed features:** Open, High, Low, Close, Volume, MA_7, MA_30, Volatility, Lag features

---

## Installation & Setup
1. Clone the repository:  
```bash
git clone <your_repo_url>
cd crypto-project
