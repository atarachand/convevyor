import time
import json
import random
import numpy as np
from datetime import datetime
import paho.mqtt.client as mqtt

def generate_sensor_data(fault=None):
    data = {
        "timestamp": datetime.now().isoformat(),
        "vibration": np.random.normal(3.0, 0.5),
        "temperature": np.random.normal(40.0, 2.0),
        "load": random.uniform(60, 100),
        "speed": np.random.normal(1.2, 0.1)
    }

    if fault == 'overheat':
        data["temperature"] += np.random.normal(20, 5)
    elif fault == 'overload':
        data["load"] += random.uniform(20, 40)
    elif fault == 'imbalance':
        data["vibration"] += np.random.normal(4.0, 1.0)
    elif fault == 'slip':
        data["speed"] -= np.random.normal(0.5, 0.2)

    return data

def start_publishing(fault=None):
    client = mqtt.Client()
    client.connect("broker.hivemq.com", 1883)
    client.loop_start()
    while True:
        payload = generate_sensor_data(fault)
        print("Publishing:", payload)
        client.publish("conveyor/data", json.dumps(payload))
        time.sleep(1)

if __name__ == "__main__":
    start_publishing(fault=None)