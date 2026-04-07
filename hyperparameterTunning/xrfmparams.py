import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
import optuna
from optuna.samplers import TPESampler
import torch
from xrfm import xRFM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def _infer_task_and_metric(y: pd.Series) -> tuple[str, str]:
    if (
        isinstance(y.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(y)
        or pd.api.types.is_string_dtype(y)
        or pd.api.types.is_bool_dtype(y)
    ):
        return "categorical", "accuracy"
    return "regression", "mse"

def _objective(trial: optuna.trial.Trial, X: pd.DataFrame, y: pd.Series, time_limit_s: int, folds: int = 5):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task_type, tuning_metric = _infer_task_and_metric(y)
    y_processed = y.copy()
    if task_type == "categorical":
        y_processed = pd.Series(pd.Categorical(y).codes, index=y.index)

    # Search space for xRFM-specific hyperparameters

    bandwidth = trial.suggest_float("bandwidth", 1, 200, log=True)
    exponent =  trial.suggest_float("exponent", 0.7, 1.4)
    norm_p = trial.suggest_float("norm_p", exponent, exponent + 0.8*(2-exponent))
    reg = trial.suggest_float("reg", 1e-6, 10, log=True)
    subset_prop = trial.suggest_float("subset_prop", 0.01, 0.5) # from 1% to 50% of the data in each leaf

    params = {
        # min subset size as a proportion of the data (to prevent overfitting and ensure enough samples in each leaf)
        "max_leaf_size": int(subset_prop * len(X)),
        "use_temperature_tuning": False,
        "time_limit_s": time_limit_s,
        "rfm_params": {
            "model": {
                "kernel": "lpq",
                "bandwidth": np.log(bandwidth),
                "norm_p": norm_p,
                "exponent": exponent,
            },
            "fit": {
                "reg": reg,
                "iters": 0,
                "return_best_params": True,
            },
        },
    }

    # run kfold cross validation on the training data
    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    result = 0
    for step, (train_index, val_index) in enumerate(kf.split(X)):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y_processed.iloc[train_index], y_processed.iloc[val_index]

        X_train_arr = X_train_fold.to_numpy(dtype=np.float32)
        X_val_arr = X_val_fold.to_numpy(dtype=np.float32)
        y_train_arr = y_train_fold.to_numpy()
        y_val_arr = y_val_fold.to_numpy()

        # Initialize and train model
        model = xRFM(**params, device=device, tuning_metric=tuning_metric)
        model.fit(X_train_arr, y_train_arr, X_val_arr, y_val_arr)
        
        # Evaluate performance
        preds = model.predict(X_val_arr)

        if task_type == "regression":
            preds_array = np.asarray(preds).reshape(-1)
            mse = mean_squared_error(y_val_arr.reshape(-1), preds_array)
            result += mse
        else:
            preds_array = np.asarray(preds)
            if preds_array.ndim > 1:
                preds_array = np.argmax(preds_array, axis=1)
            accuracy = np.mean(preds_array.reshape(-1) == y_val_arr.reshape(-1))
            result += accuracy
        
        current_average = result / (step + 1)
        trial.report(current_average, step)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return result / kf.get_n_splits()

def tunexrfm(X: pd.DataFrame, y: pd.Series, n_trials: int = 50, timeout_iteration: int = 50, timeout_s: int | None = None, folds: int = 5):
    """Tune xRFM hyperparameters using Optuna

    Args:
        X (pd.DataFrame): Training features
        y (pd.Series): Training targets
        n_trials (int, optional): Number of Optuna trials. Defaults to 50.
        timeout_iteration (int, optional): Time limit for optimization in iterations. Defaults to 50.
        timeout_s (int | None, optional): Time limit for optimization in seconds. Defaults to None (no time limit).
        folds (int, optional): Number of folds for cross-validation. Defaults to 5.

    Returns:
        optuna.Study: The Optuna study object containing the results of the optimization
    """
    _, tuning_metric = _infer_task_and_metric(y)
    direction = "minimize" if tuning_metric == "mse" else "maximize"

    sampler = TPESampler(seed=42, multivariate=True)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)

    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    try:
        study.optimize(lambda trial: _objective(trial, X, y, timeout_iteration, folds), n_trials=n_trials, timeout=timeout_s)
    except KeyboardInterrupt:
        print("Optimization interrupted. Returning completed trials so far.")
    return study

if __name__ == "__main__":
    from sklearn.datasets import fetch_california_housing
    from normalizeFeatures import normalizeFeatures

    # Load the California housing dataset
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target

    # normalize the features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = normalizeFeatures(X_train, X_test)
    

    study = tunexrfm(X_train, y_train, n_trials=50, timeout_s=300)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")