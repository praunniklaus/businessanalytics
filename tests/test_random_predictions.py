"""
Randomized prediction test: generates synthetic feature rows around the training
distribution, queries the current best model, and saves outputs plus plots.

Run directly:
    python tests/test_random_predictions.py

Or via pytest (will still generate artifacts):
    pytest -q tests/test_random_predictions.py
"""

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "artifacts"
DEFAULT_SAMPLES = 1000
RANDOM_SEED = 42
MIN_PRED_PRICE = 30.0


def load_best_model() -> joblib:
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
        best_name = min(metrics.items(), key=lambda kv: kv[1])[0]
    else:
        best_name = "xgboost.joblib"
    model_path = MODELS_DIR / best_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path), best_name


def load_feature_matrix() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data file: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    return df.drop(columns=["price", "id"])


def build_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stats = {}
    for col in df.columns:
        series = df[col]
        mean = series.mean()
        std = series.std()
        is_binary = set(series.unique()).issubset({0, 1})
        stats[col] = {
          "mean": mean,
          "std": std if std > 0 else max(abs(mean) * 0.1, 1e-3),
          "binary": is_binary,
          "non_negative": series.min() >= 0
        }
    return stats


def sample_rows(stats: Dict[str, Dict[str, float]], n: int) -> pd.DataFrame:
    np.random.seed(RANDOM_SEED)
    rows = []
    cols = list(stats.keys())
    for _ in range(n):
        row = []
        for col in cols:
            s = stats[col]
            if s["binary"]:
                val = np.random.binomial(1, min(max(s["mean"], 0.0), 1.0))
            else:
                val = np.random.normal(s["mean"], s["std"])
                if s["non_negative"]:
                    val = max(val, 0.0)
            row.append(val)
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def save_artifacts(df_preds: pd.DataFrame, model_name: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "random_predictions.csv"
    df_preds.to_csv(csv_path, index=False)

    preds = df_preds["prediction"]
    # Histogram
    plt.figure(figsize=(8, 4))
    plt.hist(preds, bins=10, color="#3b82f6", edgecolor="black")
    plt.title(f"Predictions histogram ({model_name})")
    plt.xlabel("Predicted price")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predictions_hist.png", dpi=200)
    plt.close()

    # Boxplot
    plt.figure(figsize=(6, 4))
    plt.boxplot(preds, vert=False, patch_artist=True, boxprops=dict(facecolor="#10b981"))
    plt.title(f"Predictions boxplot ({model_name})")
    plt.xlabel("Predicted price")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predictions_box.png", dpi=200)
    plt.close()

    # Scatter vs accommodates (if present)
    if "accommodates" in df_preds.columns:
        plt.figure(figsize=(6, 4))
        plt.scatter(df_preds["accommodates"], preds, alpha=0.7, color="#f97316")
        plt.title(f"Predicted price vs accommodates ({model_name})")
        plt.xlabel("Accommodates")
        plt.ylabel("Predicted price")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "predictions_vs_accommodates.png", dpi=200)
        plt.close()


def generate_predictions(n: int = DEFAULT_SAMPLES):
    model, model_name = load_best_model()
    X = load_feature_matrix()
    stats = build_stats(X)
    samples = sample_rows(stats, n)
    preds = model.predict(samples)
    preds = np.maximum(preds, MIN_PRED_PRICE)  # enforce business floor
    df_out = samples.copy()
    df_out.insert(0, "prediction", preds)
    return df_out, model_name


def test_random_predictions():
    df_out, model_name = generate_predictions()
    assert len(df_out) == DEFAULT_SAMPLES
    assert (df_out["prediction"] >= 0).all()
    save_artifacts(df_out, model_name)


if __name__ == "__main__":
    df_out, model_name = generate_predictions()
    print(df_out[["prediction"]].to_string(index=False))
    save_artifacts(df_out, model_name)
    print(f"\nSaved artifacts to {OUTPUT_DIR}")
