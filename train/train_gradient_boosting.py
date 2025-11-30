import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os
import sys
import importlib.util

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

predict_model_path = os.path.join(project_root, 'predict', 'predict_model.py')
spec = importlib.util.spec_from_file_location("predict_model", predict_model_path)
predict_model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict_model_module)
predict_model = predict_model_module.predict_model

def train_gradient_boosting():
    print("Loading data...")
    data_path = os.path.join(project_root, 'data', 'data.csv')
    data = pd.read_csv(data_path)
    
    print(f"Dataset shape: {data.shape}")
    print(f"Features: {data.shape[1] - 1}")
    
    X = data.drop('price', axis=1)
    y = data['price']
    
    print("\nSplitting data into train/test (80/20) with random_state=10...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=10
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    print("\nTraining Gradient Boosting Regressor...")
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.85,
        max_features='sqrt',
        random_state=10,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    print("\nEvaluating on training set...")
    y_train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    print(f"Training RMSE: {train_rmse:.4f}")
    
    print("\nEvaluating on test set...")
    y_test_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    print(f"Test RMSE: {test_rmse:.4f}")
    
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'gradient_boosting.joblib')
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    print("\n" + "="*50)
    print("Running predict_model.py...")
    print("="*50)
    test_rmse_from_test = predict_model(model_path, X_test, y_test)
    
    print("\n" + "="*50)
    print("Training Summary:")
    print("="*50)
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test RMSE (from predict_model.py): {test_rmse_from_test:.4f}")
    print(f"Model saved to: {model_path}")

if __name__ == '__main__':
    train_gradient_boosting()

