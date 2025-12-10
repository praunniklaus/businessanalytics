# Airbnb Price Prediction Project

A machine learning project for predicting Airbnb listing prices in Berlin, Germany, with an interactive web interface featuring a heatmap visualization and price prediction tool.

## Prerequisites

- Python 3.8 or higher
- Node.js 18 or higher
- npm or yarn

## Setup

### 1. Create and Activate Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

## Running the Project

The project consists of two components that need to run simultaneously:

### Backend API (FastAPI)

In the project root directory (with virtual environment activated):

```bash
source .venv/bin/activate
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Frontend (Next.js)

In a separate terminal, navigate to the frontend directory:

```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Project Structure

```
.
├── api/                      # FastAPI backend
│   └── main.py               # Prediction endpoints
├── data/                     # Data artifacts and helpers
│   ├── raw_data.csv          # Raw scrape
│   ├── preprocessed.csv      # Cleaned data
│   ├── data.csv              # Feature-engineered data
│   ├── split.csv             # Train/val/test ids
│   ├── feature_averages.json # Neighborhood aggregates
│   ├── create_feature_averages.py # Generate neighborhood averages
│   └── create_split.py       # Build train/val/test split
├── frontend/                 # Next.js frontend
│   └── app/
│       └── analyze/          # Analysis page with prediction form
├── heatmap/                  # Map generation
│   ├── create_map.py         # Builds static property map
│   └── property_map.html     # Generated map
├── models/                   # Trained models and benchmarks
│   ├── *.joblib              # Model artifacts
│   ├── model_benchmark.png
│   └── model_benchmark_delta.png
├── preprocessing/            # Data preprocessing
│   ├── preprocessing.py      # Main pipeline
│   └── feature_eng.py        # Feature engineering helpers
├── predict/                  # Prediction utilities
│   └── predict_model.py
├── tests/                    # Checks and diagnostic plots
│   ├── plot_actual_vs_pred.py
│   └── test_random_predictions.py
├── train/                    # Model training/evaluation
│   ├── train_*.py            # Model-specific trainers
│   ├── benchmark_models.py   # Compare models
│   └── plot_model_metrics.py # Plot metric deltas
├── README.md                 # Project overview
└── requirements.txt          # Python dependencies
```

## Generating the Heatmap

To generate the static property map:

```bash
source .venv/bin/activate
python heatmap/create_map.py
```

This creates `heatmap/property_map.html` which can be opened directly in a browser.

## Documentation 

### Airbnb Price Prediction Berlin - Project Documentation

#### Introduction and Problem Statement
How can a Berlin Airbnb host price a listing optimally? New hosts lack market context; experienced hosts often rely on gut feeling and risk empty nights or underpricing. Berlin’s ~20k+ active listings make pricing competitive, with strong dependence on location, amenities, reviews, property type, and host experience. Our goal is a data-driven tool so hosts can price confidently.

#### Our Approach
We framed pricing as supervised regression: listing features → nightly price. We built a full application: a FastAPI backend serving predictions and a Next.js frontend with an interactive map so users can input details and see comparable listings.

#### Data Processing (from raw to cleaned)
- Raw intake (`data/raw_data.csv`): full scrape with many unused columns (URLs, long descriptions, profile pics), mixed types, messy categories, missing values, outliers (prices, nights, baths/beds), inconsistent licensing, and amenities/reviews as long strings.
- Preprocessing (`data/preprocessed.csv`): drop non-signal columns; standardize neighborhoods and room/property types; parse and clean numeric fields (price, beds, baths, min/max nights); filter/cap implausible values; handle missing values with sensible fills/flags; keep scrape dates for recency.
- Result: ~9.2k cleaned listings (from ~27k raw, 79 columns) focused on signals: property traits, location, host info, reviews, amenities.

#### Feature Engineering
- Outputs: `data/data.csv` and neighborhood aggregates `data/feature_averages.json`.
- Steps: encode categoricals; explode amenities into indicators; create ratios/interactions (price per night, beds/baths per guest); add neighborhood aggregates (e.g., average price per sqm); compute domain signals like distances to key landmarks (Brandenburg Gate, Alexanderplatz); quality proxies (amenity count, per-person ratios, professional-host flag).
- Final: 77 engineered features mixing numeric, encoded categorical, and aggregate signals.

#### Modeling
- Trained/compared: linear baselines (Ridge, Lasso), trees/ensembles (Decision Tree, Random Forest, sklearn Gradient Boosting, XGBoost, LightGBM, CatBoost), MLP, stacking ensemble, plus a similarity KNN model for nearest-neighbor lookups.
- Artifacts: `models/*.joblib`, `models/model_metrics.json`, benchmark plots.

#### Why CatBoost in Production
- Best-performing in benchmarks; strong with categorical features; less overfitting; fast inference for the web app.
- Tuned via RandomizedSearchCV (5-fold CV) over iterations, learning rate, depth, regularization.

#### Model Evaluation
- Held-out 20% test set.
- Ranking: CatBoost tuned > sklearn Gradient Boosting ≈ Stacking Ensemble; XGBoost/LightGBM close; linear models as baselines; MLP and plain tree weaker.
- Key drivers: room type, capacity, location (distance to center/neighborhood), review volume, bedrooms/beds.

#### Serving and Visualization (API, heatmap, frontend)
- Backend: FastAPI (`api/`) loads trained models, validates Berlin locations, exposes prediction endpoints, and serves map data.
- Heatmap: `heatmap/` builds spatial supply/price patterns (`property_map.html`).
- Frontend: Next.js (`frontend/`) with a four-step prediction form (location, property details, amenities, reviews) and interactive map showing predicted listing alongside real comps, color-coded by price tiers; includes address geocoding for ease of use.

#### Limitations and Future Work
- Current: static training snapshot; no seasonality/demand modeling; Berlin-only scope.
- Next: add time-series features, occupancy for revenue prediction, multi-city support, confidence intervals.
