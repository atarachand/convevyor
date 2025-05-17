# Conveyor Belt Digital Twin with MQTT, AI, RUL

This version includes:
- ✅ MQTT publisher & subscriber
- ✅ Streamlit dashboard with real-time data
- ✅ Autoencoder anomaly detection
- ✅ LSTM RUL prediction

## To Use

1. Start telemetry publisher in one terminal:
```bash
python telemetry_publisher.py
```

2. Start the Streamlit dashboard in another:
```bash
streamlit run dashboard_app.py
```

Broker: `broker.hivemq.com`  
Topic: `conveyor/data`

Make sure both files are running to see live updates.