import pandas as pd
import matplotlib.pyplot as plt
import os

# Create a folder to save plots
os.makedirs("../reports/figures", exist_ok=True)

# Load processed data
btc_df = pd.read_csv("../data/processed/BTC_features.csv")
eth_df = pd.read_csv("../data/processed/ETH_features.csv")

# Convert Date to datetime for better plotting
btc_df['Date'] = pd.to_datetime(btc_df['Date'])
eth_df['Date'] = pd.to_datetime(eth_df['Date'])

# Quick Info
print("BTC Data:")
print(btc_df.info())
print("\nETH Data:")
print(eth_df.info())

# Descriptive statistics
print("\nBTC Summary Stats:")
print(btc_df.describe())

print("\nETH Summary Stats:")
print(eth_df.describe())

# --- Plot 1: BTC vs ETH Closing Price ---
plt.figure(figsize=(12, 6))
plt.plot(btc_df['Date'], btc_df['Close'], label="BTC Close", color='orange')
plt.plot(eth_df['Date'], eth_df['Close'], label="ETH Close", color='blue')
plt.title("BTC vs ETH Closing Price Trend")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("../reports/figures/btc_eth_price_trend.png")
plt.show()

# --- Plot 2: Daily % Change Distribution ---
plt.figure(figsize=(8, 5))
plt.hist(btc_df['Daily_Change'], bins=50, alpha=0.6, label='BTC', color='orange')
plt.hist(eth_df['Daily_Change'], bins=50, alpha=0.6, label='ETH', color='blue')
plt.title("Daily Percentage Change Distribution")
plt.xlabel("Daily % Change")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("../reports/figures/daily_change_distribution.png")
plt.show()

# --- Plot 3: BTC Moving Averages ---
plt.figure(figsize=(12, 6))
plt.plot(btc_df['Date'], btc_df['Close'], label="BTC Close", alpha=0.5, color='gray')
plt.plot(btc_df['Date'], btc_df['MA_7'], label="BTC MA 7", color='orange')
plt.plot(btc_df['Date'], btc_df['MA_30'], label="BTC MA 30", color='red')
plt.title("BTC Price with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("../reports/figures/btc_moving_averages.png")
plt.show()

# --- Extra Insight: Correlation between BTC & ETH ---
correlation = btc_df['Daily_Change'].corr(eth_df['Daily_Change'])
print(f"\n🔗 Correlation between BTC & ETH Daily % Change: {correlation:.2f}")
