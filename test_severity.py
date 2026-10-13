"""
Test suite for anomaly severity calculation in dashboard.py.
Verifies normal readings, moderate anomalies, and extreme anomalies across all severity tiers.
"""
import sqlite3
import numpy as np
import pandas as pd
from dashboard import load_model, detect_anomalies, fetch_latest_data

def test_anomaly_scoring():
    model, scaler = load_model()

    test_cases = pd.DataFrame([
        # Normal readings (expected: anomaly == False, anomaly_score < 0.50)
        {'case': 'Typical Normal', 'temperature': 22.0, 'pressure': 1013.0, 'vibration': 0.50, 'humidity': 45.0},
        {'case': 'Mildly Elevated Normal', 'temperature': 23.5, 'pressure': 1015.0, 'vibration': 0.58, 'humidity': 48.0},
        
        # Mild / Boundary anomalies (expected: anomaly == True, anomaly_score in [0.50, 0.60] -> LOW)
        {'case': 'Mild Anomaly (Boundary)', 'temperature': 24.0, 'pressure': 1021.0, 'vibration': 0.76, 'humidity': 54.0},
        
        # Moderate anomalies (expected: anomaly == True, anomaly_score in (0.60, 0.75] -> MEDIUM)
        {'case': 'Moderate Anomaly (Pressure Spike)', 'temperature': 22.0, 'pressure': 1032.0, 'vibration': 0.65, 'humidity': 50.0},
        {'case': 'Moderate Anomaly (Vibration+Temp)', 'temperature': 27.5, 'pressure': 1022.0, 'vibration': 0.85, 'humidity': 52.0},
        
        # Extreme anomalies (expected: anomaly == True, anomaly_score > 0.75 -> HIGH)
        {'case': 'Extreme Anomaly (Multi-Metric Spike)', 'temperature': 42.0, 'pressure': 1038.0, 'vibration': 1.10, 'humidity': 68.0},
        {'case': 'Extreme Outlier (Severe Spikes)', 'temperature': 55.0, 'pressure': 1050.0, 'vibration': 2.00, 'humidity': 85.0},
    ])

    scored = detect_anomalies(test_cases.copy(), model, scaler)
    scored['severity'] = scored.apply(
        lambda r: 'NORMAL' if not r['anomaly'] else ('HIGH' if r['anomaly_score'] > 0.75 else 'MEDIUM' if r['anomaly_score'] > 0.60 else 'LOW'),
        axis=1
    )

    print("\n--- BENCHMARK RESULTS ---")
    for _, row in scored.iterrows():
        print(f"{row['case']:38s} | Anomaly: {str(row['anomaly']):5s} | Score: {row['anomaly_score']:.4f} | Severity: {row['severity']}")

    normals = scored[~scored['anomaly']]
    assert (normals['anomaly_score'] < 0.50).all(), "Normal readings must have score < 0.50"

    anomalies = scored[scored['anomaly']]
    assert (anomalies['anomaly_score'] >= 0.50).all(), "Anomalies must have score >= 0.50"

    assert (anomalies['severity'] == 'LOW').any(), "Expected at least one LOW severity anomaly"
    assert (anomalies['severity'] == 'MEDIUM').any(), "Expected at least one MEDIUM severity anomaly"
    assert (anomalies['severity'] == 'HIGH').any(), "Expected at least one HIGH severity anomaly"

    print("\n[SUCCESS] Synthetic benchmark tests passed!")
    return scored

def test_database_distribution():
    model, scaler = load_model()
    conn = sqlite3.connect('data/sensor_stream.db')
    df = pd.read_sql('SELECT * FROM sensor_readings', conn)
    
    scored_db = detect_anomalies(df.copy(), model, scaler)
    anom_db = scored_db[scored_db['anomaly']].copy()
    anom_db['severity'] = anom_db['anomaly_score'].apply(
        lambda s: 'HIGH' if s > 0.75 else 'MEDIUM' if s > 0.60 else 'LOW'
    )
    
    counts = anom_db['severity'].value_counts()
    print("\n--- DATABASE REAL-TIME LOG DISTRIBUTION ---")
    print(f"Total readings in DB: {len(df)}")
    print(f"Total detected anomalies: {len(anom_db)}")
    for sev in ['LOW', 'MEDIUM', 'HIGH']:
        c = counts.get(sev, 0)
        pct = (c / len(anom_db) * 100) if len(anom_db) > 0 else 0
        print(f"  {sev:6s}: {c:4d} ({pct:5.1f}%)")
        
    assert counts.get('LOW', 0) > 0, "Expected LOW anomalies in real database"
    assert counts.get('MEDIUM', 0) > 0, "Expected MEDIUM anomalies in real database"
    assert counts.get('HIGH', 0) > 0, "Expected HIGH anomalies in real database"
    print("\n[SUCCESS] Real database severity distribution test passed!")

def test_dashboard_window_sample():
    model, scaler = load_model()
    conn = sqlite3.connect('data/sensor_stream.db')
    df = fetch_latest_data(conn, 300)
    df = detect_anomalies(df, model, scaler)

    anomalies = df[df['anomaly']].copy()
    anomalies['severity'] = anomalies['anomaly_score'].apply(
        lambda s: 'HIGH' if s > 0.75 else 'MEDIUM' if s > 0.60 else 'LOW'
    )

    print("\n--- DASHBOARD LATEST 300 READINGS WINDOW ---")
    print(f"Latest window readings: {len(df)}")
    print(f"Latest window anomalies: {len(anomalies)}")
    print(anomalies['severity'].value_counts().to_string())

    cols = ['timestamp', 'sensor_id', 'temperature', 'pressure', 'vibration', 'humidity', 'anomaly_score', 'severity']
    print("\nSamples per severity level:")
    for sev in ['HIGH', 'MEDIUM', 'LOW']:
        subset = anomalies[anomalies['severity'] == sev].head(2)
        print(f"\n[{sev}]")
        print(subset[cols].to_string())

if __name__ == '__main__':
    test_anomaly_scoring()
    test_database_distribution()
    test_dashboard_window_sample()
