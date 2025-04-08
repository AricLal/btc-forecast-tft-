# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from sklearn.preprocessing import StandardScaler
import datetime
import os

# ====================
# Page Config
# ====================
st.set_page_config(page_title="BTC Price Forecast", layout="wide")
st.title("📈 Bitcoin 7-Day Price Forecast")
st.markdown("Powered by a Temporal Fusion Transformer trained on BTC indicators")

# ====================
# Load data
# ====================
data_path = "bitcoin_prices.csv"
df = pd.read_csv(data_path, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
df["group_id"] = "BTC"
df["time_idx"] = (df["timestamp"] - df["timestamp"].min()).dt.days

# Indicators
df["SMA_7"] = df["Price"].rolling(7, min_periods=1).mean()
df["SMA_30"] = df["Price"].rolling(30, min_periods=1).mean()
df["RSI"] = df["Price"].pct_change().rolling(14, min_periods=1).mean()
df["weekday"] = df["timestamp"].dt.day_name()
df["target"] = df["Price"]
df = df.dropna(subset=["SMA_7", "SMA_30", "RSI", "target"])

# Scale
target_scaler = StandardScaler()
df["target_scaled"] = target_scaler.fit_transform(df[["target"]])

# Backtest split
cutoff = int(len(df) * 0.8)
train_df = df.iloc[:cutoff]
val_df = df.iloc[cutoff:]

# ====================
# Define dataset
# ====================
encoder_length = min(60, len(val_df)-10)
dataset = TimeSeriesDataSet(
    train_df,
    time_idx="time_idx",
    target="target_scaled",
    group_ids=["group_id"],
    max_encoder_length=encoder_length,
    max_prediction_length=7,
    time_varying_known_reals=["time_idx", "SMA_7", "SMA_30"],
    time_varying_known_categoricals=["weekday"],
    time_varying_unknown_reals=["target_scaled", "RSI"],
    add_relative_time_idx=True,
    add_target_scales=True,
    allow_missing_timesteps=True,
    target_normalizer=None
)

val_dataset = TimeSeriesDataSet.from_dataset(dataset, val_df, predict=True)
val_loader = val_dataset.to_dataloader(train=False, batch_size=32)

# ====================
# Load model (you can update path if saved checkpoint exists)
# ====================
@st.cache_resource
def load_model():
    return TemporalFusionTransformer.load_from_checkpoint("tft_model.ckpt")
model = load_model()

# ====================
# Predict
# ====================
predictions = model.predict(val_loader, mode="raw", return_x=True, return_y=True)
y_pred = predictions.output[0][:, :, 1]  # median forecast
y_true = predictions.y[0]  # actual

# Most recent sequence
y_pred_np = y_pred[-1].detach().cpu().numpy()
y_true_np = y_true[-1].detach().cpu().numpy()

# Inverse transform
predicted_prices = target_scaler.inverse_transform(y_pred_np.reshape(-1, 1)).flatten()
actual_prices = target_scaler.inverse_transform(y_true_np.reshape(-1, 1)).flatten()

# ====================
# Plot + Date Labels
# ====================
# Get last 7 timestamps for x-axis
future_dates = val_df["timestamp"].iloc[-7:].dt.strftime("%b %d").tolist()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(actual_prices, label="Actual")
ax.plot(predicted_prices, label="Predicted", linestyle="--")
ax.set_title("BTC Price Forecast (7-Day Horizon)")
ax.set_xlabel("Date")
ax.set_ylabel("BTC Price ($)")
ax.set_xticks(range(7))
ax.set_xticklabels(future_dates, rotation=45)
ax.grid(True)
ax.legend()
st.pyplot(fig)

# ====================
# Export Button w/ Error Metrics
# ====================
# Compute errors
abs_error = np.abs(actual_prices - predicted_prices)
pct_error = np.abs((actual_prices - predicted_prices) / actual_prices) * 100

# Create exportable dataframe
forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Price": predicted_prices,
    "Actual_Price": actual_prices,
    "Absolute_Error": abs_error,
    "Percentage_Error (%)": pct_error
})

csv = forecast_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Forecast with Errors as CSV",
    data=csv,
    file_name="btc_7_day_forecast.csv",
    mime="text/csv"
)

# ====================
# Footer
# ====================
st.markdown("---")
st.markdown("Made by Aric Lal • Model: Temporal Fusion Transformer • Data: BTC Price History")
