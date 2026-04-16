import time

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from xgboost import XGBRegressor
from xrfm import xRFM


if __name__ == "__main__":
    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))

    def target_function(X):
        return torch.randn(X.shape[0], 2, device=X.device)
    
    num_runs = 10
    # Accumulate times: dict mapping sample size -> list of times across runs
    xrfm_all_times = {}
    xgboost_all_times = {}

    xgb_n_estimators = 2000
    early_stopping_rounds = 50
    xgb_device = "cuda"
    xgboost_params = {
        "eta": 0.1,
        "n_estimators": xgb_n_estimators,
        "gamma": 0,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 1,
        "colsample_bytree": 1,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "tree_method": "hist",
        "early_stopping_rounds": early_stopping_rounds,
        "device": xgb_device,
        "random_state": 42,
    }

    leaf_rfm_params = {
        "model": {
            "kernel": "lpq",
            "bandwidth": np.log(10),
            "norm_p": 1.0,
            "exponent": 1.0,
        },
        "fit": {
            "reg": 0.1,
            "iters": 0,
            "return_best_params": True,
            "verbose": False,
        },
    }

    split_rfm_params = {
        "model": leaf_rfm_params["model"],
        "fit": {
            "get_agop_best_model": True,
            "return_best_params": False,
            "reg": 0.1,
            "iters": 0,
            "early_stop_rfm": False,
            "verbose": False,
        },
    }

    

    # Warmup: absorb CUDA initialization overhead before timing
    device = torch.device('cuda')
    X_warmup = torch.randn(100, 100, device=device)
    y_warmup = target_function(X_warmup)
    XGBRegressor(**xgboost_params).fit(X_warmup, y_warmup, eval_set=[(X_warmup, y_warmup)], verbose=False)

    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        for i in range(20):

            # Generate data
            n_samples = 1000*i + 1000
            n_features = 100
            X = torch.randn(n_samples, n_features, device=device)
            y = target_function(X)

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            xrfm_params = {
                # min subset size as a proportion of the data (to prevent overfitting and ensure enough samples in each leaf)
                "max_leaf_size": int(0.4 * n_samples),
                "use_temperature_tuning": True, # we turned this off for validation to speed it up, but we will turn it on for the final training
                "rfm_params": leaf_rfm_params,
                "default_rfm_params": split_rfm_params,
            }

            
            xrfm = xRFM(device=device, tuning_metric='mse')
            xgboost = XGBRegressor(**xgboost_params)

            start_time = time.perf_counter()
            xrfm.fit(X_train, y_train, X_val, y_val)
            end_time = time.perf_counter()
            training_time = end_time - start_time
            xrfm_all_times.setdefault(n_samples, []).append(training_time)

            start_time = time.perf_counter()
            xgboost.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            end_time = time.perf_counter()
            training_time = end_time - start_time
            xgboost_all_times.setdefault(n_samples, []).append(training_time)
    
    print(f"\nAveraged over {num_runs} runs:")
    for s in sorted(xrfm_all_times.keys()):
        avg = np.mean(xrfm_all_times[s])
        std = np.std(xrfm_all_times[s])
        print(f"xRFM training time for {s} samples: {avg:.2f} ± {std:.2f} seconds")
    
    for s in sorted(xgboost_all_times.keys()):
        avg = np.mean(xgboost_all_times[s])
        std = np.std(xgboost_all_times[s])
        print(f"XGBoost training time for {s} samples: {avg:.2f} ± {std:.2f} seconds")

    # Tab-separated format for Google Sheets
    print("\n--- Copy below into Google Sheets ---")
    print("Samples\txRFM Avg (s)\txRFM Std (s)\tXGBoost Avg (s)\tXGBoost Std (s)")
    for s in sorted(xrfm_all_times.keys()):
        xrfm_avg = np.mean(xrfm_all_times[s])
        xrfm_std = np.std(xrfm_all_times[s])
        xgb_avg = np.mean(xgboost_all_times[s])
        xgb_std = np.std(xgboost_all_times[s])
        print(f"{s}\t{xrfm_avg:.4f}\t{xrfm_std:.4f}\t{xgb_avg:.4f}\t{xgb_std:.4f}")
