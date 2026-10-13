# Real-Time IoT Sensor Dashboard with ML Anomaly Detection

A live monitoring dashboard that ingests simulated IoT sensor data in real-time and applies an **Isolation Forest** machine learning model to detect anomalies. Built with Streamlit, scikit-learn, Plotly, and SQLite.

## Project Structure

```
realtime-dashboard/
├── data_stream.py          # Simulated IoT sensor data generator (background process)
├── generate_sample_data.py # Generates historical training data
├── train_model.py          # Trains Isolation Forest and saves model + scaler
├── dashboard.py            # Streamlit real-time monitoring dashboard
├── requirements.txt        # Python dependencies
├── data/                   # Data directory (gitignored)
│   ├── historical_data.csv # Training dataset
│   └── sensor_stream.db    # Live sensor data (SQLite)
└── model/                  # Model directory (gitignored)
    ├── isolation_forest.pkl
    └── scaler.pkl
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Historical Training Data

```bash
python generate_sample_data.py
```

This creates `data/historical_data.csv` with 10,000 sensor readings (~5% anomalies).

### 3. Train the Anomaly Detection Model

```bash
python train_model.py
```

Trains an Isolation Forest model and saves it to `model/isolation_forest.pkl`.

### 4. Start the Data Stream

```bash
python data_stream.py
```

This simulates 5 IoT sensors publishing readings every 2 seconds to `data/sensor_stream.db`.
Leave this running in a separate terminal.

### 5. Launch the Dashboard

```bash
streamlit run dashboard.py
```

Opens the live dashboard in your browser. It auto-refreshes every 2 seconds.

## Dashboard Features

### Four Visualization Tabs

1. **Time Series** - Line charts for temperature, pressure, vibration, and humidity over time with anomaly markers (red X)
2. **Anomaly Analysis** - Scatter plot of anomaly scores with severity classification and statistics
3. **Anomaly Heatmap** - Pivot table heatmap showing anomaly concentration by sensor and time window
4. **Event Log** - Tabular view of all anomaly events with severity ratings (HIGH/MEDIUM/LOW)

### Real-Time Components

- **Gauge Charts** - Four gauges (one per metric) with green/yellow/red zones and live values
- **Alert System** - WARNING banner when 3+ anomalies detected within a 60-second rolling window
- **Auto-Refresh** - Dashboard polls the database every 2 seconds for new sensor data
- **Sidebar Controls** - Filter by individual sensor, adjust the number of data points displayed
- **Live Stats** - Real-time counts of total readings, anomalies detected, and latest reading timestamp

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Data Generator  │────>│  SQLite DB       │<────│  Dashboard       │
│  (data_stream.py)│     │  sensor_stream.db│     │  dashboard.py    │
└─────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                          │
┌─────────────────┐     ┌──────────────────┐              │ loads
│  Training Data   │────>│  Isolation Forest│              ▼
│  (historical)    │     │  Model           │     Anomaly Detection
└─────────────────┘     └──────────────────┘
```

### How It Works

1. **Data Generation** - Simulated IoT sensors produce readings with occasional anomalies (8% chance per reading)
2. **Training** - Isolation Forest learns the normal distribution of sensor metrics (temperature, pressure, vibration, humidity)
3. **Detection** - Each incoming reading is scored; anomalies get negative decision function values
4. **Visualization** - Plotly charts render live data, anomaly highlights, and heatmap aggregations
5. **Alerting** - A rolling 60-second window tracks anomaly counts and triggers a WARNING at 3+ events

## Key Technical Decisions

| Decision | Why |
|----------|-----|
| Isolation Forest | Unsupervised, no labeled data needed, handles multivariate anomalies |
| SQLite | Lightweight, no infrastructure required, good for demo/prototype |
| Streamlit | Zero-config UI, auto-refresh with `st.rerun()`, native Plotly integration |
| `st.cache_resource` | Model loaded once, not on every dashboard refresh |
| Sigmoid score transform | Converts decision function to 0-1 range for intuitive severity levels |

## Interview Talking Points

### 1. Why Isolation Forest for Anomaly Detection?
Isolation Forest works by isolating observations using random splits. Anomalies are easier to isolate (require fewer splits) and thus have shorter path lengths. It's computationally efficient (O(n log n)), unsupervised (no labeled data needed), and handles high-dimensional data well. For a production system you'd compare it against One-Class SVM, Autoencoders, and DBSCAN.

### 2. How Would You Scale This to Real IoT Sensors?
Replace the data generator with an ingestion layer (Kafka, Kinesis, or MQTT broker) that receives real telemetry. Use a stream processing framework like Apache Flink or Spark Structured Streaming. Store time-series data in InfluxDB or TimescaleDB instead of SQLite. Deploy the model with a serving framework (KServe, Seldon) behind the stream processor for real-time inference.

### 3. How Do You Handle Concept Drift?
If sensors degrade or operating conditions change, the model's notion of "normal" becomes stale. Solutions: (1) Periodic retraining on recent data with a sliding window, (2) Online learning variants of Isolation Forest, (3) Monitor model drift itself by tracking the distribution of anomaly scores over time and triggering retraining when the score distribution shifts.

### 4. What's the Alert Threshold Strategy?
The alert triggers at 3+ anomalies in 60 seconds. In production, this should be data-driven: analyze historical alert precision and recall across different thresholds, account for sensor-specific baselines (some sensors naturally spike more), and implement alert fatigue mitigation (escalation, cooldown periods, correlation across sensors).

### 5. How Would You Validate the Model Without Ground Truth?
Unsupervised anomaly detection is hard to validate. Strategies include: (1) Inject known synthetic anomalies and measure detection rate, (2) Use human-in-the-loop labeling for discovered anomalies, (3) Cross-validate with domain experts who know failure modes, (4) Track false positive rate by monitoring how often alerts lead to confirmed issues.

### 6. What's the End-to-End Data Flow?
Sensor -> ingestion layer -> time-series DB -> feature extraction -> ML scoring -> alert engine -> dashboard. Each component has failure modes: network partitions cause lost readings, the model can become stale, the DB can grow unbounded. A production system needs dead-letter queues, model registry + versioned deployments, data retention policies, and health checks at every hop.
