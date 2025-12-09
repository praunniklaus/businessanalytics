"""
Visualize prediction vs. actual prices and how errors grow for higher-priced listings.

Produces:
  - tests/artifacts/actual_vs_pred.png    (scatter with y=x reference)
  - tests/artifacts/error_vs_price.png    (absolute error vs actual price)
Also prints basic error stats to stdout.

Run:
  python tests/plot_actual_vs_pred.py
"""

import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "artifacts"
MIN_PRED_PRICE = 30.0


def load_best_model():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
        best_name = min(metrics.items(), key=lambda kv: kv[1])[0]
    else:
        best_name = "xgboost.joblib"
    model_path = MODELS_DIR / best_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path), best_name


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data file: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    y = df["price"]
    X = df.drop(columns=["price", "id"])
    return X, y


def main():
    model, model_name = load_best_model()
    X, y = load_dataset()
    preds = model.predict(X)
    preds = np.maximum(preds, MIN_PRED_PRICE)

    abs_err = np.abs(preds - y)
    rel_err = abs_err / np.maximum(y, 1e-6)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Scatter: actual vs predicted with y=x line
    plt.figure(figsize=(7, 6))
    plt.scatter(y, preds, alpha=0.25, color="#2563eb", edgecolors="none")
    lims = [0, max(y.max(), preds.max()) * 1.05]
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.title(f"Actual vs Predicted ({model_name})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_pred.png", dpi=200)
    plt.close()

    # Error vs actual price
    plt.figure(figsize=(7, 5))
    plt.scatter(y, abs_err, alpha=0.25, color="#f97316", edgecolors="none")
    plt.xlabel("Actual price")
    plt.ylabel("Absolute error")
    plt.title(f"Error vs Actual Price ({model_name})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "error_vs_price.png", dpi=200)
    plt.close()

    print(f"Model: {model_name}")
    print(f"Count: {len(y)}")
    print(f"Abs error stats: mean={abs_err.mean():.2f}, median={np.median(abs_err):.2f}, 90p={np.percentile(abs_err, 90):.2f}")
    print(f"Rel error stats: mean={rel_err.mean()*100:.1f}%, median={np.median(rel_err)*100:.1f}%")
    print(f"Artifacts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
