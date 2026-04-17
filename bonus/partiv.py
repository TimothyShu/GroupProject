import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
import torch


if __name__ == "__main__":

    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    
    from bonus.train import train
    from bonus.compareAccuracy import compute_holdout_metrics, print_holdout_metrics
    
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    n_samples = 10000
    X = np.random.randn(n_samples, 5)
    # Rare, sharp component: X[:,2] > 2 is rare
    y = (
        2 * X[:, 0] + X[:, 1] +
        40 * (X[:, 2] > 2.2).astype(float) +  # rare, sharp jump
        2 * X[:, 3] * X[:, 4] +
        np.random.normal(0, 0.1, n_samples)
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
    y = pd.Series(y, name="target")

    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=0)

    xrfm, resxrfm = train(X_train, y_train, trials=10, same_splits=False)

    metrics = compute_holdout_metrics(xrfm, X_test, y_test, metric="mse")
    metrics_res = compute_holdout_metrics(resxrfm, X_test, y_test, metric="mse")

    print_holdout_metrics("XRM", metrics)
    print_holdout_metrics("ResXRM", metrics_res)