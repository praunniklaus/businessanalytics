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
