import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

# ================== CONFIG ==================
# Use the cleaned CSV created by clean_csv.py
CSV_PATH = "data/gold_clean.csv"
WINDOW_SIZE = 60        # past 60 days -> predict next day


def create_sequences(data, window_size):
    """
    Convert a 1D array into sequences for LSTM.
    Each X[i] = previous 'window_size' values
    Each y[i] = next value
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


def main():
    # 1. Load cleaned gold prices
    df = pd.read_csv(CSV_PATH)

    # Ensure Date column is datetime and sorted (just to be safe)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

    # Check required column
    if "Close" not in df.columns:
        raise ValueError("CSV must contain a 'Close' column.")

    # Use only Close price for now (shape: (n, 1))
    close_prices = df["Close"].values.reshape(-1, 1)

    # 2. Scale data to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_prices)

    # 3. Create LSTM sequences
    X, y = create_sequences(scaled, WINDOW_SIZE)
    # X shape: (samples, window_size, 1)
    # y shape: (samples, 1)

    # 4. Train / test split (no shuffle because it's time-series)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape :", X_test.shape)
    print("y_test shape :", y_test.shape)

    # 5. Save arrays and scaler for later use
    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)
    np.save("X_test.npy", X_test)
    np.save("y_test.npy", y_test)

    with open("scaler.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "window_size": WINDOW_SIZE}, f)

    # Also save the processed dataframe (for Streamlit plots)
    df.to_csv("gold_processed.csv", index=False)

    print("✅ Preprocessing complete.")
    print("   Saved: X_train.npy, y_train.npy, X_test.npy, y_test.npy, scaler.pkl, gold_processed.csv")


if __name__ == "__main__":
    main()
