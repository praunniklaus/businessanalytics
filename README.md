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
├── api/                    # FastAPI backend
│   └── main.py             # API endpoints for predictions
├── data/                   # Data files
│   ├── raw_data.csv        # Original raw data
│   ├── preprocessed.csv    # Preprocessed data
│   ├── data.csv           # Final feature-engineered data
│   └── split.csv          # Train/test split IDs
├── frontend/              # Next.js frontend
│   └── app/
│       └── analyze/       # Analysis page with prediction form
├── heatmap/               # Map generation scripts
│   └── create_map.py      # Creates interactive property map
├── models/                # Trained ML models
│   ├── xgboost.joblib
│   ├── catboost.joblib
│   ├── lightgbm.joblib
│   └── gradient_boosting.joblib
├── preprocessing/         # Data preprocessing scripts
│   ├── preprocessing.py  # Main preprocessing pipeline
│   └── feature_eng.py    # Feature engineering
├── predict/              # Prediction utilities
│   └── predict_model.py
├── train/                # Model training scripts
│   ├── train_xgboost.py
│   ├── train_catboost.py
│   ├── train_lightgbm.py
│   └── train_gradient_boosting.py
└── requirements.txt       # Python dependencies
```

## Generating the Heatmap

To generate the static property map:

```bash
source .venv/bin/activate
python heatmap/create_map.py
```

This creates `heatmap/property_map.html` which can be opened directly in a browser.
