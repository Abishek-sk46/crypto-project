# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# --- Streamlit Page Config ---
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
eth_feature_cols = [col for col in eth_df.columns if col not in ["Date", "Close"]]

# --- Load trained models ---
btc_model = joblib.load("../models/btc_model.pkl")
eth_model = joblib.load("../models/eth_model.pkl")

# --- Sidebar: coin selector ---
coin = st.sidebar.selectbox("Select Coin", ["BTC", "ETH"])

# --- Choose dataset & model based on coin ---
if coin == "BTC":
    df = btc_df
    feature_cols = btc_feature_cols
    model = btc_model
else:
    df = eth_df
    feature_cols = eth_feature_cols
    model = eth_model

# --- Sidebar: select date range using actual dates ---
start_date = df['Date'].min()
end_date = df['Date'].max()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(start_date, end_date),
    min_value=start_date,
    max_value=end_date
)

# Filter dataset based on selected dates
df_selected = df[(df['Date'] >= pd.to_datetime(date_range[0])) & 
                 (df['Date'] <= pd.to_datetime(date_range[1]))]

# --- Latest prediction ---
latest_features = df[feature_cols].iloc[-1:]
latest_pred = model.predict(latest_features)[0]
latest_actual = df['Close'].iloc[-1]
price_change = latest_pred - latest_actual

st.metric(
    label=f"{coin} Predicted Price (Next Day)",
    value=f"${latest_pred:,.2f}",
    delta=f"{price_change:,.2f}"
)

# --- Plot Historical Prices (Interactive) ---
st.subheader(f"{coin} Historical Prices")
fig_hist = px.line(df_selected, x="Date", y="Close", title=f"{coin} Historical Prices")
fig_hist.update_traces(line_color="blue")
fig_hist.update_layout(xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(fig_hist, use_container_width=True)

# --- Plot Actual vs Predicted ---
st.subheader(f"{coin} Predicted vs Actual Prices")
plot_df = df.copy()
plot_df["Predicted"] = model.predict(df[feature_cols])
fig_pred = px.line(plot_df, x="Date", y=["Close", "Predicted"],
                   labels={"value": "Price (USD)", "variable": "Legend"},
                   title=f"{coin} Actual vs Predicted Prices")
st.plotly_chart(fig_pred, use_container_width=True)

# --- Display Metrics ---
def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred)/y_true))*100
    return rmse, mae, mape

all_pred = model.predict(df[feature_cols])
rmse, mae, mape = calc_metrics(df['Close'], all_pred)

st.subheader(f"{coin} Model Performance")
st.table(pd.DataFrame({
    "Metric": ["RMSE", "MAE", "MAPE (%)"],
    "Value": [f"{rmse:.2f}", f"{mae:.2f}", f"{mape:.2f}%"]
}))

# --- Download Predictions ---
pred_df = df.copy()
pred_df["Predicted"] = all_pred
csv = pred_df.to_csv(index=False)
st.download_button(
    label="📥 Download Predictions as CSV",
    data=csv,
    file_name=f"{coin}_predictions.csv",
    mime="text/csv",
)
