# Conveyor Belt Digital Twin with MQTT, AI, RUL

This project simulates a real-time conveyor belt digital twin with:
- MQTT telemetry stream
- Autoencoder anomaly detection
- LSTM RUL prediction
- Streamlit visualization

## How to Run

1. Start publisher (simulated sensor)
```bash
python telemetry_publisher.py
```

2. Start dashboard
```bash
streamlit run dashboard_app.py
```

3. Or deploy dashboard to Streamlit Cloud (use `dashboard_app.py`)

Live topic: `conveyor/data` (via broker.hivemq.com)