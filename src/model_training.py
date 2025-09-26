# model_training.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# --- Load Data ---
btc_df = pd.read_csv("../data/processed/BTC_features.csv")
eth_df = pd.read_csv("../data/processed/ETH_features.csv")

# Ensure Date is datetime
btc_df['Date'] = pd.to_datetime(btc_df['Date'])
eth_df['Date'] = pd.to_datetime(eth_df['Date'])

# --- Feature Engineering: BTC Lag Features ---
btc_df['Close_lag1'] = btc_df['Close'].shift(1)
btc_df['Close_lag2'] = btc_df['Close'].shift(2)
btc_df = btc_df.dropna()  # remove NaNs due to lag

# Ensure folders exist
os.makedirs("../reports/figures", exist_ok=True)
os.makedirs("../models", exist_ok=True)

def train_and_evaluate(df, coin_name, feature_cols=None):
    """Train RandomForest model and evaluate it for the given coin."""
    # Default features if not provided
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in ["Date", "Close"]]

    X = df[feature_cols]
    y = df['Close']

    # Time-based train-test split
    split_index = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # Hyperparameter tuning for BTC only
    if coin_name == "BTC":
        param_grid = {
            'n_estimators':[100, 200, 500],
            'max_depth':[10, 20, None],
            'min_samples_split':[2, 5, 10],
            'min_samples_leaf':[1, 2, 4]
        }
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                                   cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
    else:
        # Default RandomForest for ETH
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n✅ {coin_name} Model Trained Successfully!")
    print(f"📊 {coin_name} RMSE: {rmse:.2f}")
    print(f"📈 {coin_name} R² Score: {r2:.4f}")

    # Plot Actual vs Predicted
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label="Actual", color="blue")
    plt.plot(y_pred, label="Predicted", color="orange")
    plt.title(f"{coin_name} Price Prediction (Actual vs Predicted)")
    plt.xlabel("Time (Test Set)")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"../reports/figures/{coin_name.lower()}_price_prediction.png")
    plt.close()

    # Save model
    joblib.dump(model, f"../models/{coin_name.lower()}_model.pkl")
    print(f"✅ {coin_name} model saved to ../models/{coin_name.lower()}_model.pkl")

    return model

# --- Train Models ---
btc_features = ['Open','High','Low','Volume','MA_7','MA_30','Volatility','Close_lag1','Close_lag2']
btc_model = train_and_evaluate(btc_df, "BTC", btc_features)
eth_model = train_and_evaluate(eth_df, "ETH")  # ETH default features

print("\n✅ All models trained and results saved in ../reports/figures/ and ../models/")
