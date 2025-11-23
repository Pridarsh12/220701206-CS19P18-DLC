import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import datetime as dt

from tensorflow.keras.models import load_model
from realtime_api import fetch_live_gold_price


# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Gold Price Prediction using LSTM",
    layout="wide"
)


# ------------------ CUSTOM PREMIUM BLACK + GOLD THEME ------------------
def inject_custom_css():
    st.markdown(
        """
        <style>

        /* MAIN BACKGROUND */
        .stApp {
            background-color: #000000 !important;
            color: #FFFFFF !important;
        }

        html, body, [class*="css"] {
            color: #FFFFFF !important;
            font-family: "Segoe UI", sans-serif !important;
        }

        /* TITLES */
        h1, h2, h3, h4, h5 {
            color: #FFD700 !important;
            font-weight: 700 !important;
        }

        /* SIDEBAR */
        section[data-testid="stSidebar"] {
            background-color: #0A0A0A !important;
            border-right: 1px solid #333333;
        }

        /* EXPANDER */
        details {
            background-color: #0D0D0D !important;
            border-radius: 8px !important;
            border: 1px solid #333333 !important;
        }
        summary {
            color: #FFD700 !important;
            font-size: 18px !important;
        }

        /* METRIC CARD FIXES */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1A1A1A, #222222) !important;
    border-radius: 12px !important;
    padding: 15px !important;
    border: 1px solid #FFD700AA !important;
}
div[data-testid="stMetric"] > label,
div[data-testid="stMetricLabel"] {
    color: #FFFFFF !important;
    font-size: 16px !important;
}
div[data-testid="stMetric"] > div,
div[data-testid="stMetricValue"] {
    color: #FFD700 !important;
    font-size: 26px !important;
    font-weight: 700 !important;
}


        /* BUTTON — GOLD & BLACK */
        button[kind="primary"] {
            background: linear-gradient(135deg, #FFD700, #FFA500) !important;
            color: #000000 !important;
            border-radius: 999px !important;
            padding: 10px 20px !important;
            font-weight: 700 !important;
            border: none !important;
        }

        /* SLIDER TEXT FIX */
        .stSlider > div > div > div {
            color: #FFD700 !important;
        }

        /* TAB STYLE */
        button[data-baseweb="tab"] {
            background-color: #000000 !important;
            color: #BBBBBB !important;
            border-radius: 10px 10px 0 0 !important;
            border: 1px solid #333333 !important;
            border-bottom: none !important;
            font-size: 16px !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #222222, #444444) !important;
            color: #FFD700 !important;
            border-bottom: 3px solid #FFD700 !important;
        }

        /* TABLES */
        .stDataFrame div[data-testid="stTable"] {
            background-color: #111111 !important;
            color: white !important;
        }

        /* Fix metric label color */
div[data-testid="stMetric"] div:nth-child(1),
span[data-testid="stMetricLabel"] {
    color: #FFFFFF !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}

/* Fix slider label text */
div[data-testid="stSlider"] > label {
    color: #FFFFFF !important;
    font-size: 18px !important;
    font-weight: 600 !important;
}


        /* REMOVE DEFAULT UI CLUTTER */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True
    )


inject_custom_css()



# ------------------ LOAD MODEL + DATA ------------------
@st.cache_resource
def load_artifacts():
    model = load_model("gold_lstm.h5")

    with open("scaler.pkl", "rb") as f:
        scaler_data = pickle.load(f)

    scaler = scaler_data["scaler"]
    window_size = scaler_data["window_size"]

    df = pd.read_csv("gold_processed.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    return model, scaler, window_size, df


model, scaler, WINDOW_SIZE, df = load_artifacts()

# Scale close prices
close_prices = df["Close"].values.reshape(-1, 1)
scaled_close = scaler.transform(close_prices)



# ------------------ FORECAST FUNCTION ------------------
def forecast_next_days(n_days):
    last_window = scaled_close[-WINDOW_SIZE:].copy().reshape(1, WINDOW_SIZE, 1)
    preds_scaled = []
    current = last_window

    for _ in range(n_days):
        pred = model.predict(current, verbose=0)[0][0]
        preds_scaled.append(pred)
        arr = current.reshape(WINDOW_SIZE)
        arr = np.append(arr[1:], pred)
        current = arr.reshape(1, WINDOW_SIZE, 1)

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds = scaler.inverse_transform(preds_scaled).flatten()
    return preds



# ------------------ UI TABS ------------------
st.title("🏅 Real-Time Gold Price Prediction using LSTM")

tab1, tab2, tab3 = st.tabs(["Overview", "Forecast", "Historical Data"])



# ===============================================================
# ⭐ TAB 1 — OVERVIEW
# ===============================================================
with tab1:
    st.subheader("Project Overview")

    st.write(
        """
        This dashboard uses a **Stacked LSTM Neural Network** to forecast future gold prices  
        and includes **Real-Time Gold Price** fetched from GoldAPI.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        try:
            st.image("training_loss.png")
        except:
            st.warning("training_loss.png missing. Run train_model.py first.")

    with col2:
        try:
            st.image("test_predictions.png")
        except:
            st.warning("test_predictions.png missing. Run train_model.py first.")



# ===============================================================
# ⭐ TAB 2 — FORECAST + REAL-TIME PRICE
# ===============================================================
with tab2:
    st.subheader("Real-Time Gold Price & Future Forecast")

    # ---------------- REAL TIME PRICE ----------------
    with st.expander("🔴 Live Gold Price (via GoldAPI.io)", expanded=True):
        live_price, err = fetch_live_gold_price()

        if err:
            st.error(f"Error fetching live price: {err}")
        else:
            colA, colB = st.columns(2)
            with colA:
                st.metric(
                    label="Current Gold Price (XAU/USD)",
                    value=f"{live_price:,.2f} USD/oz"
                )
            with colB:
                st.success("Last updated: " + dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            st.caption("Fetched using your GoldAPI Key.")

    st.markdown("---")

    # ---------------- FUTURE FORECAST ----------------
    st.subheader("LSTM Forecast for Future Prices")

    n = st.slider("Select forecast horizon (days):", 1, 30, 7)

    if st.button("Generate Forecast", type="primary"):
        preds = forecast_next_days(n)

        last_date = df["Date"].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n)

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Price (USD)": preds
        })

        st.write("### 📄 Predicted Prices")
        st.dataframe(forecast_df)

        # ---------- Plot ----------
        st.write("### 📈 Historical vs Forecast (with Live Price)")

        hist = df.tail(100)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(hist["Date"], hist["Close"], label="Historical Price", color="#FFD700", linewidth=2)
        ax.plot(future_dates, preds, "--", label="Predicted Price", color="#00BFFF", linewidth=2)

        # mark live price
        if live_price:
            live_point = hist["Date"].iloc[-1] + pd.Timedelta(days=1)
            ax.scatter(live_point, live_price, color="red", s=90, label="Live Price")

        ax.set_xlabel("Date")
        ax.set_ylabel("Gold Price (USD/oz)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)



# ===============================================================
# ⭐ TAB 3 — HISTORICAL DATA
# ===============================================================
with tab3:
    st.subheader("Historical Gold Price (Full Dataset)")

    st.dataframe(df.tail(25))

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(df["Date"], df["Close"], color="#FFD700")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price (USD/oz)")
    ax2.set_title("Historical Gold Prices")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
