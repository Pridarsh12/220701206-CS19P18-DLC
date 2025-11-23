import numpy as np
import pickle
import math
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error


def main():
    # 1. Load preprocessed data
    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")

    with open("scaler.pkl", "rb") as f:
        scaler_info = pickle.load(f)

    scaler = scaler_info["scaler"]
    window_size = scaler_info["window_size"]

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape :", X_test.shape)
    print("y_test shape :", y_test.shape)
    print("Window size  :", window_size)

    # 2. Build stacked LSTM model
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(window_size, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))  # predict next day's scaled Close price

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mean_squared_error"
    )

    model.summary()

    # 3. Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=20,          # you can increase later
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    # 4. Predict on test set
    y_pred = model.predict(X_test)

    # 5. Inverse scale to original price values
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)

    # 6. Calculate metrics
    rmse = math.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)

    print(f"\nTest RMSE: {rmse:.4f}")
    print(f"Test MAE : {mae:.4f}")

    # 7. Plot training vs validation loss
    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_loss.png", dpi=150)
    plt.close()

    # 8. Plot Actual vs Predicted on test set
    plt.figure()
    plt.plot(y_test_inv, label="Actual Price")
    plt.plot(y_pred_inv, label="Predicted Price")
    plt.xlabel("Time Step (Test Set)")
    plt.ylabel("Gold Price")
    plt.title("Actual vs Predicted Gold Price (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("test_predictions.png", dpi=150)
    plt.close()

    # 9. Save the trained model
    model.save("gold_lstm.h5")
    print("\nModel saved as gold_lstm.h5")
    print("Plots saved as training_loss.png and test_predictions.png")


if __name__ == "__main__":
    main()
