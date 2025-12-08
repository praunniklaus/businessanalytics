import os
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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


def train_with_tuning(X_train, y_train):
    base_model = HistGradientBoostingRegressor(
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )
    
    param_distributions = {
        "max_iter": [200, 400, 600, 800, 1000],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "max_depth": [6, 8, 10, 12, None],
        "min_samples_leaf": [10, 20, 30, 50],
        "max_leaf_nodes": [31, 50, 80, 100, None],
        "l2_regularization": [0.0, 0.1, 0.5, 1.0, 2.0],
    }
    
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="neg_root_mean_squared_error",
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    
    start_time = time.time()
    search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    return search.best_estimator_, search.best_params_, training_time


def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train: {len(X_train)}, Test: {len(X_test)}, Features: {X_train.shape[1]}")
    
    print("Training with hyperparameter tuning...")
    model, best_params, training_time = train_with_tuning(X_train, y_train)
    print(f"Best params: {best_params}")
    
    print("Evaluating...")
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    
    model_path = os.path.join(MODELS_DIR, "histgb_tuned.joblib")
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
