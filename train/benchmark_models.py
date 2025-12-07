"""
Train and benchmark multiple models on the same train/test split.
Adds cross-validated hyperparameter search and a stacking ensemble.
Prints a concise KPI table and saves each trained model to the models/ folder.
"""

import os
import sys
import time
from typing import Dict, Any, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import StackingRegressor
from sklearn.base import clone

# Third-party boosters
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "data.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RANDOM_STATE = 10
CV_FOLDS = 5
SEARCH_ITER = 6  # keep searches lightweight to avoid long runtimes

os.makedirs(MODELS_DIR, exist_ok=True)


def load_features():
    """Load dataset and split into train/test."""
    data = pd.read_csv(DATA_PATH)
    X = data.drop(["price", "id"], axis=1)
    y = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=10
    )
    return X_train, X_test, y_train, y_test


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def format_metrics(name: str, kpis: Dict[str, float], elapsed: float) -> str:
    return (
        f"{name:24s} | RMSE: {kpis['rmse']:.2f} | MAE: {kpis['mae']:.2f} | "
        f"MAPE: {kpis['mape']:.2f}% | R²: {kpis['r2']:.3f} | time: {elapsed:.1f}s"
    )


def make_transformed_regressor(base: Any) -> TransformedTargetRegressor:
    """Wrap a regressor to train on log(price) and evaluate on original scale."""
    return TransformedTargetRegressor(
        regressor=base,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def run_cv_model(
    name: str,
    estimator: Any,
    param_distributions: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    log_target: bool = False,
) -> Tuple[str, Dict[str, float], float, Any]:
    """
    Hyperparameter search with RandomizedSearchCV (RMSE) then evaluate on held-out test.
    Supports optional log-target transformation.
    """
    search_estimator = estimator
    if log_target:
        # When using TransformedTargetRegressor, parameter names are prefixed with regressor__
        search_estimator = make_transformed_regressor(estimator)
        param_distributions = {f"regressor__{k}": v for k, v in param_distributions.items()}

    search = RandomizedSearchCV(
        search_estimator,
        param_distributions=param_distributions,
        n_iter=SEARCH_ITER,
        scoring="neg_root_mean_squared_error",
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )

    print(f"--- CV search + training {name} ---")
    start = time.time()
    search.fit(X_train, y_train)
    best_estimator = search.best_estimator_
    train_time = time.time() - start

    y_pred = best_estimator.predict(X_test)
    kpis = metrics(y_test, y_pred)

    model_path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(best_estimator, model_path)
    log_note = " (log-price target)" if log_target else ""
    print(format_metrics(name, kpis, train_time) + log_note)
    print(f"Best params: {search.best_params_}")
    print(f"Saved model to {model_path}\n")

    return name, kpis, train_time, best_estimator


def main():
    print("Loading data from", DATA_PATH)
    X_train, X_test, y_train, y_test = load_features()
    print(f"Train samples: {len(X_train):,}, Test samples: {len(X_test):,}")
    print(f"Feature count: {X_train.shape[1]}")
    print("\nTraining and benchmarking models on the same split...\n")

    candidates: List[Dict[str, Any]] = [
        {
            "name": "xgboost_tuned",
            "model": XGBRegressor(
                n_estimators=800,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=2.0,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=1.0,
                gamma=0.05,
                objective="reg:squarederror",
                n_jobs=-1,
                random_state=10,
                verbosity=0,
            ),
            "log_target": False,
            "params": {
                "n_estimators": [800, 1000, 1200],
                "learning_rate": [0.03, 0.04, 0.05],
                "max_depth": [5, 6, 7],
                "min_child_weight": [1.5, 2.0, 2.5],
                "subsample": [0.85, 0.9, 0.95],
                "colsample_bytree": [0.85, 0.9, 0.95],
                "reg_alpha": [0.0, 0.05, 0.1],
                "reg_lambda": [0.9, 1.0, 1.2],
                "gamma": [0.0, 0.05, 0.1],
            },
        },
        {
            "name": "xgboost_log_enhanced",
            # Trains on log(price) for stabler loss and better fit on skewed targets
            "model": XGBRegressor(
                n_estimators=1200,
                learning_rate=0.03,
                max_depth=6,
                min_child_weight=1.5,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.05,
                reg_lambda=1.2,
                gamma=0.0,
                objective="reg:squarederror",
                n_jobs=-1,
                random_state=10,
                verbosity=0,
            ),
            "log_target": True,
            "params": {
                "n_estimators": [1000, 1200, 1400],
                "learning_rate": [0.02, 0.03, 0.04],
                "max_depth": [5, 6, 7],
                "min_child_weight": [1.0, 1.5, 2.0],
                "subsample": [0.85, 0.9, 0.95],
                "colsample_bytree": [0.85, 0.9, 0.95],
                "reg_alpha": [0.0, 0.05, 0.1],
                "reg_lambda": [0.9, 1.0, 1.2],
                "gamma": [0.0, 0.05],
            },
        },
        {
            "name": "xgboost_improved",
            "model": XGBRegressor(
                n_estimators=1400,
                learning_rate=0.025,
                max_depth=6,
                min_child_weight=1.2,
                subsample=0.92,
                colsample_bytree=0.92,
                reg_alpha=0.05,
                reg_lambda=1.1,
                gamma=0.0,
                objective="reg:squarederror",
                n_jobs=-1,
                random_state=10,
                verbosity=0,
            ),
            "log_target": False,
            "params": {
                "n_estimators": [1200, 1400, 1600],
                "learning_rate": [0.02, 0.025, 0.03],
                "max_depth": [5, 6],
                "min_child_weight": [1.0, 1.2, 1.5],
                "subsample": [0.9, 0.92, 0.95],
                "colsample_bytree": [0.9, 0.92, 0.95],
                "reg_alpha": [0.0, 0.05, 0.1],
                "reg_lambda": [1.0, 1.1, 1.2],
                "gamma": [0.0, 0.05],
            },
        },
        {
            "name": "lightgbm_tuned",
            "model": LGBMRegressor(
                n_estimators=900,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=48,
                min_child_samples=15,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=10,
                n_jobs=-1,
                verbose=-1,
            ),
            "log_target": False,
            "params": {
                "n_estimators": [700, 900, 1100],
                "learning_rate": [0.03, 0.05, 0.07],
                "max_depth": [5, 6, 7],
                "num_leaves": [48, 64, 80],
                "min_child_samples": [10, 15, 20],
                "subsample": [0.8, 0.85, 0.9],
                "colsample_bytree": [0.8, 0.85, 0.9],
                "reg_alpha": [0.0, 0.1, 0.2],
                "reg_lambda": [0.8, 1.0, 1.2],
            },
        },
        {
            "name": "catboost_tuned",
            "model": CatBoostRegressor(
                iterations=700,
                learning_rate=0.05,
                depth=8,
                l2_leaf_reg=4.0,
                loss_function="RMSE",
                random_seed=10,
                verbose=0,
                thread_count=-1,
            ),
            "log_target": False,
            "params": {
                "iterations": [600, 800, 1000],
                "learning_rate": [0.03, 0.05, 0.07],
                "depth": [6, 8, 10],
                "l2_leaf_reg": [3.0, 4.0, 5.0],
                "subsample": [0.8, 0.9, 1.0],
            },
        },
        {
            "name": "sklearn_gbr",
            "model": GradientBoostingRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=4,
                min_samples_split=4,
                min_samples_leaf=2,
                subsample=0.9,
                max_features="sqrt",
                random_state=10,
                verbose=0,
            ),
            "log_target": False,
            "params": {},
        },
    ]

    results: List[Tuple[str, Dict[str, float], float]] = []
    fitted_for_stack: List[Tuple[str, Any]] = []

    for candidate in candidates:
        name = candidate["name"]
        model = candidate["model"]
        log_target = candidate.get("log_target", False)
        params = candidate.get("params", {})

        if params:
            name, kpis, train_time, best_estimator = run_cv_model(
                name, model, params, X_train, y_train, X_test, y_test, log_target=log_target
            )
        else:
            # Baseline (no CV) to keep as lightweight comparator
            print(f"--- Training {name} ---")
            start = time.time()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            train_time = time.time() - start
            kpis = metrics(y_test, y_pred)
            model_path = os.path.join(MODELS_DIR, f"{name}.joblib")
            joblib.dump(model, model_path)
            print(format_metrics(name, kpis, train_time))
            print(f"Saved model to {model_path}\n")
            best_estimator = model

        results.append((name, kpis, train_time))
        fitted_for_stack.append((name, best_estimator))

    # Build a stacking ensemble from the best tuned models (excluding the baseline gbr if desired)
    base_for_stack = [
        (name, est)
        for name, est in fitted_for_stack
        if name in {"xgboost_improved", "catboost_tuned", "lightgbm_tuned", "xgboost_log_enhanced"}
    ]

    if base_for_stack:
        print("--- Training stacking_ensemble ---")
        meta_model = RidgeCV(alphas=[0.1, 1.0, 5.0], cv=CV_FOLDS)
        stack_reg = StackingRegressor(
            estimators=[(n, clone(e)) for n, e in base_for_stack],
            final_estimator=meta_model,
            passthrough=True,
            cv=CV_FOLDS,
            n_jobs=-1,
        )
        start = time.time()
        stack_reg.fit(X_train, y_train)
        train_time = time.time() - start
        y_pred = stack_reg.predict(X_test)
        kpis = metrics(y_test, y_pred)
        results.append(("stacking_ensemble", kpis, train_time))
        model_path = os.path.join(MODELS_DIR, "stacking_ensemble.joblib")
        joblib.dump(stack_reg, model_path)
        print(format_metrics("stacking_ensemble", kpis, train_time))
        print(f"Saved model to {model_path}\n")

    # Sort by RMSE ascending
    results.sort(key=lambda item: item[1]["rmse"])

    print("\n========= Benchmark Summary (best RMSE first) =========")
    for name, kpis, elapsed in results:
        print(format_metrics(name, kpis, elapsed))

    best_name, best_kpis, _ = results[0]
    print("\nBest model:", best_name)
    print(
        f"Best RMSE: {best_kpis['rmse']:.2f}, "
        f"MAE: {best_kpis['mae']:.2f}, "
        f"MAPE: {best_kpis['mape']:.2f}%, "
        f"R²: {best_kpis['r2']:.3f}"
    )


if __name__ == "__main__":
    # Ensure repo root on path for consistency with other scripts
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    main()
