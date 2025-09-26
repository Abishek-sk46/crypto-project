import yfinance as yf
from pathlib import Path

# Folder to save raw CSVs
DATA_DIR = Path("../data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_symbol(symbol, start="2019-01-01"):
    df = yf.download(symbol, start=start, progress=False)
    path = DATA_DIR / f"{symbol}.csv"
    df.to_csv(path)
    print(f"✅ Saved {symbol} data to {path}")
    return df

if __name__ == "__main__":
    download_symbol("BTC-USD")  # Bitcoin
    download_symbol("ETH-USD")  # Ethereum