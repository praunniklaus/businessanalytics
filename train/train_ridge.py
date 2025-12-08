import os
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "data.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RANDOM_STATE = 10

os.makedirs(MODELS_DIR, exist_ok=True)


def load_data():
    data = pd.read_csv(DATA_PATH)
    X = data.drop(["price", "id"], axis=1)
    y = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train: {len(X_train)}, Test: {len(X_test)}, Features: {X_train.shape[1]}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(random_state=RANDOM_STATE))
    ])

    param_grid = {
        "regressor__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
    }

    search = GridSearchCV(
        pipeline,
        param_grid,
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1
    )

    print("Training Ridge Regression with GridSearch...")
    start_time = time.time()
    search.fit(X_train, y_train)
    training_time = time.time() - start_time

    model = search.best_estimator_
    print(f"Best alpha: {search.best_params_['regressor__alpha']}")

    print("Evaluating...")
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)

    model_path = os.path.join(MODELS_DIR, "ridge.joblib")
    joblib.dump(model, model_path)

    print(f"\nResults:")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  MAE: {metrics['mae']:.2f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  R2: {metrics['r2']:.4f}")
    print(f"  Time: {training_time:.1f}s")
    print(f"  Saved: {model_path}")


if __name__ == "__main__":
    main()
