# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

st.set_page_config(page_title="Crypto Price Predictor", layout="wide")

st.title("📈 Crypto Price Predictor")
st.markdown("Predicting BTC & ETH prices using Random Forest models.")

# --- Load Data ---
btc_df = pd.read_csv("../data/processed/BTC_features.csv")
eth_df = pd.read_csv("../data/processed/ETH_features.csv")

# Convert Date to datetime
btc_df['Date'] = pd.to_datetime(btc_df['Date'])
eth_df['Date'] = pd.to_datetime(eth_df['Date'])

# --- Add lag features for BTC (same as training) ---
btc_df['Close_lag1'] = btc_df['Close'].shift(1)
btc_df['Close_lag2'] = btc_df['Close'].shift(2)
btc_df = btc_df.dropna().reset_index(drop=True)

# --- Define feature columns ---
btc_feature_cols = ['Open','High','Low','Volume','MA_7','MA_30','Volatility','Close_lag1','Close_lag2']
# For ETH, use all columns except Date and Close (same as training)
eth_feature_cols = [col for col in eth_df.columns if col not in ["Date", "Close"]]

# --- Load trained models ---
btc_model = joblib.load("../models/btc_model.pkl")
eth_model = joblib.load("../models/eth_model.pkl")

# --- Sidebar: coin selector ---
coin = st.sidebar.selectbox("Select Coin", ["BTC", "ETH"])
date_range = st.sidebar.slider("Select Date Range", 0, len(btc_df)-1, (0, len(btc_df)-1))

# --- Prepare features & prediction ---
if coin == "BTC":
    df = btc_df
    feature_cols = btc_feature_cols
    model = btc_model
else:
    df = eth_df
    feature_cols = eth_feature_cols
    model = eth_model

# Historical data
df_selected = df.iloc[date_range[0]:date_range[1]+1]

# Latest prediction (most recent day)
latest_features = df[feature_cols].iloc[-1:] 
latest_pred = model.predict(latest_features)[0]

st.subheader(f"{coin} Latest Predicted Price: ${latest_pred:,.2f}")

# --- Plot Historical Prices ---
st.subheader(f"{coin} Historical Prices")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_selected['Date'], df_selected['Close'], label="Actual Price", color='blue')
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# --- Plot Actual vs Predicted on Entire Dataset ---
st.subheader(f"{coin} Predicted vs Actual Prices")

# Predict on all data
all_features = df[feature_cols]
all_pred = model.predict(all_features)

fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(df['Date'], df['Close'], label="Actual Price", color='blue')
ax2.plot(df['Date'], all_pred, label="Predicted Price", color='orange', alpha=0.7)
ax2.set_xlabel("Date")
ax2.set_ylabel("Price (USD)")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# --- Display Metrics ---
def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred)/y_true))*100
    return rmse, mae, mape

rmse, mae, mape = calc_metrics(df['Close'], all_pred)
st.markdown(f"**{coin} Metrics:** RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")
