import streamlit as st
import pandas as pd
import plotly.express as px
from telemetry_simulation import stream_data
from ai_models import build_autoencoder, build_lstm, prepare_sequences
from sklearn.preprocessing import MinMaxScaler
import numpy as np

st.set_page_config(page_title="AI Conveyor Belt Dashboard", layout="wide")
st.title("🛰️ Conveyor Belt Digital Twin with AI Monitoring")

# Sidebar Controls
fault = st.sidebar.selectbox("Inject Fault", [None, "overheat", "overload", "imbalance", "slip"])

# Simulate Data
df = stream_data(200, fault)
st.subheader("Live Telemetry")
st.dataframe(df.tail(10), use_container_width=True)

# Visualization
fig = px.line(df, x='timestamp', y=['vibration', 'temperature', 'load', 'speed'], title="Sensor Trends")
st.plotly_chart(fig, use_container_width=True)

# Preprocessing
features = df[['vibration', 'temperature', 'load', 'speed']]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(features)

# Autoencoder for Anomaly Detection
autoencoder = build_autoencoder(X_scaled.shape[1])
autoencoder.fit(X_scaled, X_scaled, epochs=10, batch_size=16, verbose=0)
recon = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - recon, 2), axis=1)
threshold = np.percentile(mse, 95)
df['Anomaly'] = mse > threshold

# Show Anomalies
st.subheader("🔴 Anomaly Detection")
fig2 = px.scatter(df, x='timestamp', y='vibration', color=df['Anomaly'].astype(str),
                  title="Vibration with Anomalies")
st.plotly_chart(fig2, use_container_width=True)

# LSTM for RUL Prediction
n_steps = 10
seq_data = X_scaled[:, 0]  # Assume vibration signal
X_seq, y_seq = prepare_sequences(seq_data, n_steps)
X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))

lstm_model = build_lstm((n_steps, 1))
lstm_model.fit(X_seq, y_seq, epochs=10, verbose=0)
rul_pred = lstm_model.predict(X_seq[-1].reshape(1, n_steps, 1))

# Show RUL
st.subheader("⏳ Remaining Useful Life Prediction")
st.metric("Predicted RUL (vibration)", f"{rul_pred[0][0]*100:.2f} cycles")

# Alerts
if fault:
    st.error(f"⚠️ FAULT DETECTED: {fault.upper()} INJECTED!")
if df['Anomaly'].iloc[-1]:
    st.warning("🚨 Anomaly Detected in Latest Data!")