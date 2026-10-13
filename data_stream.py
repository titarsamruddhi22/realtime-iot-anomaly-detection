"""
Simulated IoT sensor data stream generator.
Produces readings every 2 seconds and stores them in SQLite.
"""

import os
import time
import random
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "sensor_stream.db")

# Normal operating ranges
SENSOR_PROFILES = {
    "sensor_1": {"temperature": (22, 1.5), "pressure": (1013, 3), "vibration": (0.5, 0.08), "humidity": (45, 4)},
    "sensor_2": {"temperature": (23, 2.0), "pressure": (1015, 4), "vibration": (0.6, 0.1), "humidity": (50, 5)},
    "sensor_3": {"temperature": (21, 1.0), "pressure": (1010, 3), "vibration": (0.4, 0.06), "humidity": (40, 3)},
    "sensor_4": {"temperature": (24, 2.5), "pressure": (1020, 5), "vibration": (0.7, 0.12), "humidity": (55, 6)},
    "sensor_5": {"temperature": (20, 1.2), "pressure": (1008, 2), "vibration": (0.3, 0.05), "humidity": (38, 3)},
}


def init_db():
    """Create the sensor readings table if it doesn't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            sensor_id TEXT NOT NULL,
            temperature REAL,
            pressure REAL,
            vibration REAL,
            humidity REAL
        )
    """)
    conn.commit()
    return conn


def generate_reading():
    """Generate a single sensor reading with occasional anomalies."""
    sensor_id = random.choice(list(SENSOR_PROFILES.keys()))
    profile = SENSOR_PROFILES[sensor_id]

    # 8% chance of anomaly (spikes in one metric)
    anomaly = random.random() < 0.08

    metrics = {}
    for metric, (mean, std) in profile.items():
        value = random.gauss(mean, std)
        if anomaly and metric == random.choice(["temperature", "pressure", "vibration", "humidity"]):
            # Spike the metric 3-5 standard deviations
            value += random.choice([-1, 1]) * random.uniform(3, 5) * std
        metrics[metric] = round(value, 4)

    return {
        "timestamp": datetime.now().isoformat(),
        "sensor_id": sensor_id,
        **metrics,
    }


def stream_data(interval=2, max_readings=1000):
    """Continuously generate sensor readings and store in SQLite."""
    conn = init_db()
    count = 0

    # Seed some initial data so dashboard isn't empty
    for _ in range(20):
        reading = generate_reading()
        conn.execute(
            "INSERT INTO sensor_readings (timestamp, sensor_id, temperature, pressure, vibration, humidity) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (reading["timestamp"], reading["sensor_id"],
             reading["temperature"], reading["pressure"],
             reading["vibration"], reading["humidity"]),
        )
    conn.commit()

    while count < max_readings:
        reading = generate_reading()
        conn.execute(
            "INSERT INTO sensor_readings (timestamp, sensor_id, temperature, pressure, vibration, humidity) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (reading["timestamp"], reading["sensor_id"],
             reading["temperature"], reading["pressure"],
             reading["vibration"], reading["humidity"]),
        )
        conn.commit()
        count += 1
        print(f"[{reading['timestamp']}] {reading['sensor_id']}: "
              f"T={reading['temperature']:.1f} P={reading['pressure']:.1f} "
              f"V={reading['vibration']:.3f} H={reading['humidity']:.1f}")
        time.sleep(interval)

    conn.close()


if __name__ == "__main__":
    stream_data(interval=2, max_readings=1000)
