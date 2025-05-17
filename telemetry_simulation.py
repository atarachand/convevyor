import pandas as pd
import numpy as np
import random
from datetime import datetime

def generate_sensor_data(fault=None):
    timestamp = datetime.now()

    vibration = np.random.normal(3.0, 0.5)
    temperature = np.random.normal(40.0, 2.0)
    load = random.uniform(60, 100)
    speed = np.random.normal(1.2, 0.1)

    if fault == 'overheat':
        temperature += np.random.normal(20, 5)
    elif fault == 'overload':
        load += random.uniform(20, 40)
    elif fault == 'imbalance':
        vibration += np.random.normal(4.0, 1.0)
    elif fault == 'slip':
        speed -= np.random.normal(0.5, 0.2)

    return {
        "timestamp": timestamp,
        "vibration": round(vibration, 2),
        "temperature": round(temperature, 2),
        "load": round(load, 2),
        "speed": round(speed, 2)
    }

def stream_data(n=100, fault=None):
    return pd.DataFrame([generate_sensor_data(fault) for _ in range(n)])