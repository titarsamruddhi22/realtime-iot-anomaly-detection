"""
Real-Time IoT Sensor Monitoring Dashboard with ML Anomaly Detection.

Run the data stream first (in a separate terminal):
    python data_stream.py

Then run this dashboard:
    streamlit run dashboard.py

The dashboard auto-refreshes every 2 seconds and shows:
- Live sensor readings (line charts)
- Anomaly detection results with anomaly scores
- Alert system (3+ anomalies in 60s = WARNING)
- Statistics panel per sensor
- Anomaly event log table
"""

import os
import pickle
import sqlite3
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "data", "sensor_stream.db")
MODEL_PATH = os.path.join(BASE, "model", "isolation_forest.pkl")
SCALER_PATH = os.path.join(BASE, "model", "scaler.pkl")

METRICS = ["temperature", "pressure", "vibration", "humidity"]
UNITS = {"temperature": "\u00b0C", "pressure": "hPa", "vibration": "mm/s", "humidity": "%"}


@st.cache_resource
def load_model():
    """Load trained Isolation Forest and scaler."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def get_db():
    """Get SQLite connection."""
    if not os.path.exists(DB_PATH):
        st.error("Database not found. Run `python data_stream.py` first to start generating data.")
        return None
    return sqlite3.connect(DB_PATH)


def fetch_latest_data(conn, n=300):
    """Fetch the latest N readings."""
    df = pd.read_sql(
        "SELECT * FROM sensor_readings ORDER BY id DESC LIMIT ?",
        conn,
        params=[n],
    )
    if df.empty:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    return df


def detect_anomalies(df, model, scaler):
    """Run anomaly detection and add anomaly flags + scores."""
    if df.empty:
        df["anomaly"] = False
        df["anomaly_score"] = 0.0
        return df

    X = df[METRICS].values
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)

    df["anomaly"] = predictions == -1
    # Scale the decision function (typically in [-0.25, 0.25]) to a 0-1 anomaly score.
    # A scaling factor of 15 maps decision_function == 0 to 0.50, mild anomalies to
    # 0.50-0.60 (LOW), moderate anomalies to 0.60-0.75 (MEDIUM), and severe to >0.75 (HIGH).
    scaled_scores = np.clip(15.0 * scores, -50.0, 50.0)
    df["anomaly_score"] = 1.0 / (1.0 + np.exp(scaled_scores))

    return df


def check_alert_window(df, window_sec=60, threshold=3):
    """True if there are `threshold` or more anomalies in the last `window_sec`."""
    if df.empty:
        return False, 0
    now = df["timestamp"].max()
    window_start = now - timedelta(seconds=window_sec)
    recent = df[df["timestamp"] >= window_start]
    count = int(recent["anomaly"].sum())
    return count >= threshold, count


def build_gauge(value, min_val, max_val, title, unit, thresholds):
    """Plotly gauge for a single metric."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 2),
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"{title} ({unit})", "font": {"size": 14}},
        number={"font": {"size": 20}},
        gauge={
            "axis": {"range": [min_val, max_val]},
            "bar": {"color": "#2E86AB"},
            "steps": [
                {"range": [min_val, thresholds[0]], "color": "#d1ecf1"},
                {"range": [thresholds[0], thresholds[1]], "color": "#fff3cd"},
                {"range": [thresholds[1], max_val], "color": "#f8d7da"},
            ],
        },
    ))
    fig.update_layout(height=180, margin={"l": 10, "r": 10, "t": 40, "b": 10})
    return fig


# === Page Config ===
st.set_page_config(page_title="IoT Sensor Monitor", page_icon="\U0001f4e1", layout="wide")
st.title("\U0001f4e1 Real-Time IoT Sensor Dashboard")
st.caption(f"ML-powered anomaly detection on live sensor data \u2022 Refreshes every {2}s")

# === Load Model ===
try:
    model, scaler = load_model()
except FileNotFoundError:
    st.error("Model files not found. Run `python train_model.py` first.")
    st.stop()

# === Load Data ===
conn = get_db()
if conn is None:
    st.stop()

df = fetch_latest_data(conn, 300)

if df.empty:
    st.info("No sensor data yet. Start the data stream with `python data_stream.py`.")
    st.stop()

# === Anomaly Detection ===
df = detect_anomalies(df, model, scaler)

# === Alert Check ===
alert_active, alert_count = check_alert_window(df, window_sec=60, threshold=3)
if alert_active:
    st.error(
        f"\u26a0\ufe0f ALERT \u2014 {alert_count} anomalies detected in the last 60 seconds! "
        "Immediate investigation recommended."
    )

# === Sidebar Controls ===
with st.sidebar:
    st.header("Controls")
    selected_sensor = st.selectbox("Filter by Sensor", ["All"] + sorted(df["sensor_id"].unique().tolist()))
    display_points = st.slider("Data Points", 50, 300, 150)

    st.divider()
    st.header("Live Stats")
    anomaly_total = int(df["anomaly"].sum())
    total_points = len(df)

    if "prev_total" not in st.session_state:
        st.session_state["prev_total"] = total_points

    st.metric("Total Readings", total_points)
    st.metric("Anomalies Detected", f"{anomaly_total} ({anomaly_total / total_points * 100:.1f}%)")

    latest_ts = df["timestamp"].max()
    st.metric("Latest Reading", latest_ts.strftime("%H:%M:%S"))

    st.session_state["prev_total"] = total_points

# Filter by sensor if selected
if selected_sensor != "All":
    df_view = df[df["sensor_id"] == selected_sensor]
else:
    df_view = df

# === Gauge Charts Row ===
latest = df_view.iloc[-1]

gauge_ranges = {
    "temperature": (10, 50, 28, 35),
    "pressure": (980, 1050, 1022, 1030),
    "vibration": (0, 3, 0.8, 1.2),
    "humidity": (10, 80, 60, 70),
}

cols = st.columns(4)
for i, metric in enumerate(METRICS):
    r = gauge_ranges[metric]
    with cols[i]:
        fig = build_gauge(
            latest[metric],
            r[0], r[1],
            metric.capitalize(),
            UNITS[metric],
            thresholds=(r[2], r[3]),
        )
        st.plotly_chart(fig, use_container_width=True)

# === Tab Layout ===
tab_ts, tab_anomaly, tab_heatmap, tab_log = st.tabs(
    ["\U0001f4c8 Time Series", "\U0001f50d Anomaly Analysis", "\U0001f525 Anomaly Heatmap", "\U0001f4cb Event Log"],
)

with tab_ts:
    st.subheader("Sensor Readings Over Time")
    display_data = df_view.tail(display_points)
    for metric in METRICS:
        fig = px.line(
            display_data,
            x="timestamp",
            y=metric,
            color="sensor_id" if selected_sensor == "All" else None,
            title=f"{metric.capitalize()} ({UNITS[metric]})",
            template="plotly_white",
        )
        anomalies = display_data[display_data["anomaly"]]
        if not anomalies.empty:
            fig.add_scatter(
                x=anomalies["timestamp"],
                y=anomalies[metric],
                mode="markers",
                marker={"color": "red", "size": 10, "symbol": "x"},
                name="Anomaly",
            )
        fig.update_layout(height=280, legend={"orientation": "h", "y": -0.3})
        st.plotly_chart(fig, use_container_width=True)

with tab_anomaly:
    st.subheader("Anomaly Detection Results")
    if df_view["anomaly"].any():
        anomaly_df = df_view[df_view["anomaly"]]
        st.write(f"**{len(anomaly_df)} anomalies** found in the last {len(df_view)} readings.")

        fig = px.scatter(
            df_view.tail(display_points),
            x="timestamp",
            y="anomaly_score",
            color="anomaly",
            title="Anomaly Scores Over Time",
            color_discrete_map={True: "red", False: "steelblue"},
            template="plotly_white",
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Latest Anomaly Score", f"{df_view.iloc[-1]['anomaly_score']:.3f}")
        col_b.metric("Max Anomaly Score", f"{df_view['anomaly_score'].max():.3f}")
        col_c.metric("Mean Anomaly Score", f"{df_view['anomaly_score'].mean():.3f}")
    else:
        st.success("\u2705 No anomalies detected in current data window.")

with tab_heatmap:
    st.subheader("Anomaly Concentration Heatmap")

    df_hm = df_view.copy()
    df_hm["time_bin"] = df_hm["timestamp"].dt.floor("min")

    if not df_hm.empty:
        anomaly_pivot = df_hm.pivot_table(
            index="sensor_id",
            columns="time_bin",
            values="anomaly",
            aggfunc="sum",
            fill_value=0,
        )

        if not anomaly_pivot.empty and (anomaly_pivot > 0).any().any():
            fig = px.imshow(
                anomaly_pivot,
                labels={"x": "Time", "y": "Sensor", "color": "Anomalies"},
                color_continuous_scale="YlOrRd",
                aspect="auto",
                title="Anomaly Count by Sensor and Time",
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No anomalies to visualize in the current data window.")

    col1, col2 = st.columns(2)
    with col1:
        sensor_counts = df_view[df_view["anomaly"]]["sensor_id"].value_counts().reset_index()
        sensor_counts.columns = ["sensor_id", "count"]
        if not sensor_counts.empty:
            fig = px.bar(
                sensor_counts,
                x="sensor_id",
                y="count",
                title="Anomalies by Sensor",
                color="count",
                color_continuous_scale="Reds",
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No anomalies detected in current view.")
    with col2:
        fig = px.histogram(
            df_view,
            x="anomaly_score",
            nbins=30,
            title="Distribution of Anomaly Scores",
            color_discrete_sequence=["#2E86AB"],
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

with tab_log:
    st.subheader("Anomaly Event Log")
    anomaly_events = df_view[df_view["anomaly"]].copy()
    if not anomaly_events.empty:
        anomaly_events["severity"] = anomaly_events["anomaly_score"].apply(
            lambda s: "\U0001f534 HIGH" if s > 0.75 else "\U0001f7e0 MEDIUM" if s > 0.6 else "\U0001f7e1 LOW"
        )
        display_cols = ["timestamp", "sensor_id", "temperature", "pressure", "vibration", "humidity", "anomaly_score", "severity"]
        st.dataframe(
            anomaly_events[display_cols].sort_values("timestamp", ascending=False).head(100),
            use_container_width=True,
            hide_index=True,
        )
        st.write(f"*Showing {len(anomaly_events)} anomaly events (most recent first)*")
    else:
        st.success("\u2705 No anomalies to display.")

# === Auto Refresh ===
st.caption("\U0001f504 Dashboard auto-refreshes to show new sensor data as it arrives.")
time.sleep(2)
st.rerun()
