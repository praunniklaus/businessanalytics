import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import os

def predict_model(model_path, X_test=None, y_test=None):
    print(f"\nLoading model from {model_path}...")
    model = joblib.load(model_path)
    
    if X_test is None or y_test is None:
        print("Loading test data from data/data.csv...")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(project_root, 'data', 'data.csv')
        data = pd.read_csv(data_path)
        X = data.drop('price', axis=1)
        y = data['price']
        
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=10
        )
    
    print(f"Test set size: {X_test.shape[0]}")
    
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\nTest RMSE: {rmse:.4f}")
    
    print(f"\nTest Results Summary:")
    print(f"  Number of test samples: {len(y_test)}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Mean actual price: {y_test.mean():.2f}")
    print(f"  Mean predicted price: {y_pred.mean():.2f}")
    
    return rmse

if __name__ == '__main__':
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, 'models')
    
    if len(sys.argv) > 1:
        model_arg = sys.argv[1]
        if os.path.isabs(model_arg) or os.path.exists(model_arg):
            model_path = model_arg
        elif model_arg.endswith('.joblib'):
            model_path = os.path.join(models_dir, model_arg)
        else:
            model_path = os.path.join(models_dir, f"{model_arg}.joblib")
    else:
        default_model = 'gradient_boosting.joblib'
        model_path = os.path.join(models_dir, default_model)
        print(f"No model specified. Using default: {default_model}")
        print(f"Usage: python predict/predict_model.py <model_name>")
        print(f"Example: python predict/predict_model.py gradient_boosting")
        print(f"         python predict/predict_model.py models/gradient_boosting.joblib")
        print()
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print(f"Available models in {models_dir}:")
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                if f.endswith('.joblib'):
                    print(f"  - {f}")
        sys.exit(1)
    
    predict_model(model_path)

