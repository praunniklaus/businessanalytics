import json
import os
import pandas as pd


def safe_mean(series, default):
    if series is None:
        return default
    series = series.dropna()
    return float(series.mean()) if not series.empty else default


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "data.csv")
    raw_path = os.path.join(project_root, "data", "raw_data.csv")
    out_path = os.path.join(project_root, "data", "feature_averages.json")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Expected training data at {data_path}")

    df = pd.read_csv(data_path)

    # Pull num_listings from raw_data if available
    num_listings_series = None
    if os.path.exists(raw_path):
        raw_df = pd.read_csv(raw_path)
        if "host_total_listings_count" in raw_df.columns:
            num_listings_series = raw_df["host_total_listings_count"]

    averages = {
        "host_experience_days": safe_mean(df.get("host_experience"), 365.0),
        "num_listings": safe_mean(num_listings_series, 1.0),
        "review_scores_rating": safe_mean(df.get("review_scores_rating"), 4.7),
        "review_scores_accuracy": safe_mean(df.get("review_scores_accuracy"), 4.7),
        "review_scores_cleanliness": safe_mean(df.get("review_scores_cleanliness"), 4.7),
        "review_scores_checkin": safe_mean(df.get("review_scores_checkin"), 4.7),
        "review_scores_communication": safe_mean(df.get("review_scores_communication"), 4.7),
        "review_scores_location": safe_mean(df.get("review_scores_location"), 4.7),
        "review_scores_value": safe_mean(df.get("review_scores_value"), 4.7),
        "number_of_reviews": safe_mean(df.get("number_of_reviews"), 50.0),
        "bedrooms": safe_mean(df.get("bedrooms"), 1.0),
        "beds": safe_mean(df.get("beds"), 1.0),
        "bathrooms": safe_mean(df.get("bathrooms"), 1.0),
    }

    with open(out_path, "w") as f:
        json.dump(averages, f, indent=2)

    print(f"Wrote averages to {out_path}")


if __name__ == "__main__":
    main()
