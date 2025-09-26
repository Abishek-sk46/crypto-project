import pandas as pd
from pathlib import Path

RAW_DIR = Path("../data/raw")
PROCESSED_DIR = Path("../data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def clean_data(file_path):
    # Skip first 2 rows which are extra headers
    df = pd.read_csv(file_path, skiprows=2)
    
    # Rename columns if needed
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Convert numerical columns to float
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing values
    df = df.dropna().reset_index(drop=True)
    
    return df

# Clean BTC and ETH
btc_df = clean_data(RAW_DIR / "BTC-USD.csv")
eth_df = clean_data(RAW_DIR / "ETH-USD.csv")

# Save cleaned files
btc_df.to_csv(PROCESSED_DIR / "BTC_clean.csv", index=False)
eth_df.to_csv(PROCESSED_DIR / "ETH_clean.csv", index=False)

print("✅ BTC and ETH data cleaned and saved to data/processed/")


def add_features(df):
    df['Daily_Change'] = df['Close'].pct_change() * 100
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()
    df['Volatility'] = (df['High'] - df['Low']) / df['Open'] * 100
    df = df.dropna().reset_index(drop=True)
    return df

# Add features
btc_df = add_features(btc_df)
eth_df = add_features(eth_df)

# Save final processed data
btc_df.to_csv(PROCESSED_DIR / "BTC_features.csv", index=False)
eth_df.to_csv(PROCESSED_DIR / "ETH_features.csv", index=False)

print("✅ Feature engineering completed and saved to data/processed/")
