import streamlit as st
import pandas as pd
import json
import plotly.express as px
import numpy as np
from collections import deque
from ai_models import build_autoencoder, build_lstm, prepare_sequences
from sklearn.preprocessing import MinMaxScaler
import paho.mqtt.client as mqtt
import threading

st.set_page_config(page_title="MQTT Conveyor Belt Digital Twin", layout="wide")
st.title("🛰️ Conveyor Belt Digital Twin with MQTT Streaming")

# Global data buffer
data_buffer = deque(maxlen=200)

# MQTT Callbacks
def on_connect(client, userdata, flags, rc):
    client.subscribe("conveyor/data")

def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    data_buffer.append(payload)

# Start MQTT subscriber thread
def start_mqtt_listener():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect("broker.hivemq.com", 1883)
    client.loop_start()

threading.Thread(target=start_mqtt_listener, daemon=True).start()

# Streamlit UI
st.sidebar.markdown("### Status: Listening to MQTT stream...")
st.sidebar.write("Live topic: `conveyor/data`")

if len(data_buffer) > 10:
    df = pd.DataFrame(data_buffer)
    st.subheader("📡 Live Telemetry Stream")
    st.dataframe(df.tail(10), use_container_width=True)

    fig = px.line(df, x='timestamp', y=['vibration', 'temperature', 'load', 'speed'], title="Sensor Telemetry Trends")
    st.plotly_chart(fig, use_container_width=True)

    # Preprocess and scale
    features = df[['vibration', 'temperature', 'load', 'speed']]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(features)

    # Autoencoder
    autoencoder = build_autoencoder(X_scaled.shape[1])
    autoencoder.fit(X_scaled, X_scaled, epochs=5, batch_size=16, verbose=0)
    recon = autoencoder.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - recon, 2), axis=1)
    threshold = np.percentile(mse, 95)
    df['Anomaly'] = mse > threshold

    st.subheader("🔴 Anomaly Detection")
    fig2 = px.scatter(df, x='timestamp', y='vibration', color=df['Anomaly'].astype(str),
                      title="Vibration with Anomalies")
    st.plotly_chart(fig2, use_container_width=True)

    # LSTM
    n_steps = 10
    seq_data = X_scaled[:, 0]
    X_seq, y_seq = prepare_sequences(seq_data, n_steps)
    X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))
    lstm_model = build_lstm((n_steps, 1))
    lstm_model.fit(X_seq, y_seq, epochs=5, verbose=0)
    pred = lstm_model.predict(X_seq[-1].reshape(1, n_steps, 1))

    st.subheader("⏳ Remaining Useful Life Prediction")
    st.metric("RUL (vibration)", f"{pred[0][0]*100:.2f} cycles")

    if df['Anomaly'].iloc[-1]:
        st.warning("🚨 Anomaly Detected in Latest Data!")
else:
    st.info("Waiting for data from MQTT broker...")