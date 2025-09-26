# evaluation.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import joblib
import os

# Ensure output folder exists
os.makedirs("../reports/figures", exist_ok=True)

# --- Load processed data ---
btc_df = pd.read_csv("../data/processed/BTC_features.csv")
eth_df = pd.read_csv("../data/processed/ETH_features.csv")

# Convert Date to datetime
btc_df['Date'] = pd.to_datetime(btc_df['Date'])
eth_df['Date'] = pd.to_datetime(eth_df['Date'])

# --- Add BTC lag features (same as training) ---
btc_df['Close_lag1'] = btc_df['Close'].shift(1)
btc_df['Close_lag2'] = btc_df['Close'].shift(2)
btc_df = btc_df.dropna().reset_index(drop=True)

# Load trained models
btc_model = joblib.load("../models/btc_model.pkl")
eth_model = joblib.load("../models/eth_model.pkl")

# --- Prepare features in the exact same order as training ---
btc_feature_cols = ['Open', 'High', 'Low', 'Volume', 'MA_7', 'MA_30', 'Volatility', 'Close_lag1', 'Close_lag2']
# For ETH, use all columns except Date and Close (same as training)
eth_feature_cols = [col for col in eth_df.columns if col not in ["Date", "Close"]]

X_btc = btc_df[btc_feature_cols]
y_btc = btc_df["Close"]

X_eth = eth_df[eth_feature_cols]
y_eth = eth_df["Close"]

# --- Predictions ---
btc_pred = btc_model.predict(X_btc)
eth_pred = eth_model.predict(X_eth)

# --- Metrics function ---
def calc_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return rmse, mae, mape

# --- BTC Metrics ---
btc_rmse, btc_mae, btc_mape = calc_metrics(y_btc, btc_pred)
print("BTC Metrics:")
print(f"RMSE: {btc_rmse:.2f}, MAE: {btc_mae:.2f}, MAPE: {btc_mape:.2f}%")

# --- ETH Metrics ---
eth_rmse, eth_mae, eth_mape = calc_metrics(y_eth, eth_pred)
print(f"\nETH Metrics:")
print(f"RMSE: {eth_rmse:.2f}, MAE: {eth_mae:.2f}, MAPE: {eth_mape:.2f}%")

# --- Generate comparison plots ---
# BTC Prediction vs Actual plot
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(y_btc.values[-100:], label="Actual BTC", color="blue")
plt.plot(btc_pred[-100:], label="Predicted BTC", color="orange")
plt.title("BTC: Actual vs Predicted (Last 100 days)")
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid()

# ETH Prediction vs Actual plot
plt.subplot(1, 2, 2)
plt.plot(y_eth.values[-100:], label="Actual ETH", color="green")
plt.plot(eth_pred[-100:], label="Predicted ETH", color="red")
plt.title("ETH: Actual vs Predicted (Last 100 days)")
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("../reports/figures/btc_pred_vs_actual.png", dpi=300, bbox_inches='tight')
plt.close()

# Save individual ETH plot
plt.figure(figsize=(12, 6))
plt.plot(y_eth.values[-100:], label="Actual ETH", color="green")
plt.plot(eth_pred[-100:], label="Predicted ETH", color="red")
plt.title("ETH: Actual vs Predicted (Last 100 days)")
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("../reports/figures/eth_pred_vs_actual.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ Evaluation complete!")
print(f"📊 Prediction vs Actual plots saved to ../reports/figures/")

