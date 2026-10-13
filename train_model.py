"""Train anomaly detection model using Isolation Forest and save it."""

import os
import pickle
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "isolation_forest.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "model", "scaler.pkl")

FEATURES = ["temperature", "pressure", "vibration", "humidity"]


def train_model(data_path=None):
    """Train Isolation Forest on historical data and save model + scaler."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), "data", "historical_data.csv")

    # Load data
    df = pd.read_csv(data_path)
    X = df[FEATURES].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Isolation Forest
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        max_samples="auto",
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # Save model and scaler
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    # Evaluate on training data
    predictions = model.predict(X_scaled)
    n_anomalies = (predictions == -1).sum()
    print(f"Model trained on {len(df)} samples")
    print(f"Detected {n_anomalies} anomalies in training data ({n_anomalies/len(df)*100:.1f}%)")
    print(f"Model saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")
    return model, scaler


def load_model():
    """Load trained model and scaler from disk."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


if __name__ == "__main__":
    train_model()
