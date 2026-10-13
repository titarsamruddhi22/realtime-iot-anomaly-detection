"""Generate historical sensor data for training the anomaly detection model."""

import os
import random
import csv
from datetime import datetime, timedelta

def generate_historical_data(n_rows=10000, output_path="data/historical_data.csv"):
    """Generate realistic IoT sensor data with ~5% anomalies injected."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    random.seed(42)
    start_time = datetime(2024, 1, 1)
    sensor_ids = [f"sensor_{i}" for i in range(1, 6)]

    # Normal ranges for each metric
    normal = {
        "temperature": (20.0, 25.0),   # mean, std
        "pressure": (1013.0, 5.0),      # hPa
        "vibration": (0.5, 0.1),        # mm/s
        "humidity": (45.0, 5.0),        # %
    }

    rows = []
    for i in range(n_rows):
        timestamp = start_time + timedelta(seconds=i * 10)
        sensor_id = random.choice(sensor_ids)

        is_anomaly = random.random() < 0.05  # 5% anomalies

        if is_anomaly:
            # Inject anomalies: values 3-5 std deviations from normal
            temp_mean, temp_std = normal["temperature"]
            pressure_mean, pressure_std = normal["pressure"]
            vib_mean, vib_std = normal["vibration"]
            hum_mean, hum_std = normal["humidity"]

            # Randomly choose which metric to anomaly
            anomaly_metric = random.choice(["temperature", "pressure", "vibration", "humidity"])

            temperature = temp_mean + random.gauss(0, temp_std)
            pressure = pressure_mean + random.gauss(0, pressure_std)
            vibration = vib_mean + random.gauss(0, vib_std)
            humidity = hum_mean + random.gauss(0, hum_std)

            if anomaly_metric == "temperature":
                temperature = temp_mean + random.choice([-1, 1]) * random.uniform(3, 5) * temp_std
            elif anomaly_metric == "pressure":
                pressure = pressure_mean + random.choice([-1, 1]) * random.uniform(3, 5) * pressure_std
            elif anomaly_metric == "vibration":
                vibration = vib_mean + random.uniform(2, 4) * vib_std  # always higher
            elif anomaly_metric == "humidity":
                humidity = hum_mean + random.choice([-1, 1]) * random.uniform(3, 5) * hum_std
        else:
            # Normal values
            temp_mean, temp_std = normal["temperature"]
            pressure_mean, pressure_std = normal["pressure"]
            vib_mean, vib_std = normal["vibration"]
            hum_mean, hum_std = normal["humidity"]

            temperature = temp_mean + random.gauss(0, temp_std)
            pressure = pressure_mean + random.gauss(0, pressure_std)
            vibration = abs(vib_mean + random.gauss(0, vib_std))
            humidity = hum_mean + random.gauss(0, hum_std)

        rows.append({
            "timestamp": timestamp.isoformat(),
            "sensor_id": sensor_id,
            "temperature": round(temperature, 2),
            "pressure": round(pressure, 2),
            "vibration": round(abs(vibration), 4),
            "humidity": round(humidity, 2),
        })

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "sensor_id", "temperature", "pressure", "vibration", "humidity"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {n_rows} historical records -> {output_path}")
    print(f"Anomaly rate ~5% ({int(n_rows * 0.05)} expected anomalies)")
    return output_path

if __name__ == "__main__":
    generate_historical_data()
