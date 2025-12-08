import os
import re
import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def parse_bathrooms(series):
    return series.apply(
        lambda x: float(re.findall(r"\d+\.?\d*", str(x))[0])
        if pd.notna(x) and re.findall(r"\d+\.?\d*", str(x))
        else None
    )


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(project_root, "data", "raw_data.csv")
    out_path = os.path.join(project_root, "models", "knn_similarity.joblib")

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Expected raw data at {raw_path}")

    raw_df = pd.read_csv(raw_path)

    raw_df["price_clean"] = (
        raw_df["price"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    raw_df["bathrooms_numeric"] = parse_bathrooms(raw_df.get("bathrooms_text"))

    feature_cols = [
        "latitude",
        "longitude",
        "accommodates",
        "bedrooms",
        "beds",
        "bathrooms_numeric",
        "price_clean",
    ]

    feature_df = raw_df[["id"] + feature_cols].dropna()

    if feature_df.empty:
        raise ValueError("No rows available after cleaning for KNN training.")

    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df[feature_cols])

    n_neighbors = min(50, len(feature_df))
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
    nn.fit(X)

    bundle = {
        "nn": nn,
        "scaler": scaler,
        "features": feature_df.reset_index(drop=True),
        "feature_names": feature_cols,
    }

    joblib.dump(bundle, out_path)
    print(f"Saved KNN similarity model to {out_path} using {len(feature_df)} listings.")


if __name__ == "__main__":
    main()
