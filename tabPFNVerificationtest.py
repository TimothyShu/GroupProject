from dotenv import load_dotenv
import numpy as np
import os
from tabpfn import TabPFNClassifier, TabPFNRegressor
import torch

from hyperparameterTunning.utils import infer_task_and_metric, process_categorical_target


def _load_tabpfn_token():
    load_dotenv() 
    
    return os.getenv("TABPFN_TOKEN")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))

    from normalizeFeatures import normalizeFeatures
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    token = _load_tabpfn_token()
    if not token:
        raise RuntimeError(
            "TABPFN_TOKEN is not set. Add it to .env in the project root or set it in PowerShell:\n"
            "TABPFN_TOKEN=your_token_here\n"
            'or\n$env:TABPFN_TOKEN = "<your-token>"'
        )
    # Load the California housing dataset
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target

    task_type, _ = infer_task_and_metric(y)
    y = process_categorical_target(y)

    # normalize the features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, X_test = normalizeFeatures(X_train, X_test)

    max_train_samples = 1000
    if len(X_train) > max_train_samples:
        X_train = X_train.iloc[:max_train_samples]
        y_train = y_train.iloc[:max_train_samples]

    X_train_arr = X_train.to_numpy(dtype=np.float32)
    X_test_arr = X_test.to_numpy(dtype=np.float32)

    if task_type == "categorical":
        y_train_arr = np.asarray(y_train).reshape(-1)
        y_test_arr = np.asarray(y_test).reshape(-1)
        tabPFN = TabPFNClassifier(device="cuda" if torch.cuda.is_available() else "cpu")
    else:
        y_train_arr = np.asarray(y_train, dtype=np.float32).reshape(-1)
        y_test_arr = np.asarray(y_test, dtype=np.float32).reshape(-1)
        tabPFN = TabPFNRegressor(device="cuda" if torch.cuda.is_available() else "cpu")

    print(f"task={task_type}, X_train={X_train_arr.shape}, y_train={y_train_arr.shape}")
    tabPFN.fit(X_train_arr, y_train_arr)
    preds = tabPFN.predict(X_test_arr)
    print(f"pred_shape={np.asarray(preds).shape}, y_test_shape={y_test_arr.shape}")

    # Evaluate the predictions
    if task_type == "categorical":
        accuracy = np.mean(np.asarray(preds).reshape(-1) == y_test_arr.reshape(-1))
        print(f"Test Accuracy: {accuracy:.4f}")
    else:
        mse = np.mean((np.asarray(preds).reshape(-1) - y_test_arr.reshape(-1)) ** 2)
        print(f"Test MSE: {mse:.4f}")

    