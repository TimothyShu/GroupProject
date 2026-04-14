import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBClassifier, XGBRegressor
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from utils import infer_task_and_metric, process_categorical_target
    

def _objective(trial: optuna.trial.Trial, X: pd.DataFrame, y: pd.Series, folds: int = 5):
    _, tuning_metric = infer_task_and_metric(y)

    xgb_device = "cuda" if torch.cuda.is_available() else "cpu"

    params = {
        "eta": trial.suggest_float("eta", 1e-2, 3e-1, log=True),
        "n_estimators": 5000,
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "max_delta_step": trial.suggest_float("max_delta_step", 0.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
        "early_stopping_rounds": 50,
        "tree_method": "hist",
        "device": xgb_device,
        "random_state": 42,
    }

    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    result = 0.0

    for step, (train_index, val_index) in enumerate(kf.split(X)):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        X_train_arr = X_train_fold.to_numpy(dtype=np.float32)
        X_val_arr = X_val_fold.to_numpy(dtype=np.float32)
        y_train_arr = y_train_fold.to_numpy()
        y_val_arr = y_val_fold.to_numpy()

        if tuning_metric == "mse":
            model = XGBRegressor(
                objective="reg:squarederror",
                eval_metric="rmse",
                **params,
            )
        else:
            n_classes = int(np.max(y.to_numpy())) + 1
            if n_classes <= 2:
                model = XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    **params,
                )
            else:
                model = XGBClassifier(
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    num_class=n_classes,
                    **params,
                )

        model.fit(X_train_arr, y_train_arr, eval_set=[(X_val_arr, y_val_arr)], verbose=False)
        preds = model.predict(X_val_arr)

        if tuning_metric == "mse":
            score = mean_squared_error(y_val_arr.reshape(-1), np.asarray(preds).reshape(-1))
        else:
            score = np.mean(np.asarray(preds).reshape(-1) == y_val_arr.reshape(-1))

        result += score

        current_average = result / (step + 1)
        trial.report(current_average, step)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return result / kf.get_n_splits()


def tunexgboost(X: pd.DataFrame, y: pd.Series, timeout_s: int | None = None, folds: int = 5):
    _, tuning_metric = infer_task_and_metric(y)
    direction = "minimize" if tuning_metric == "mse" else "maximize"

    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)

    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    try:
        study.optimize(lambda trial: _objective(trial, X, y, folds), timeout=timeout_s)
    except KeyboardInterrupt:
        print("Optimization interrupted. Returning completed trials so far.")
    return study


if __name__ == "__main__":
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target

    y = process_categorical_target(y)

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    study = tunexgboost(X_train, y_train, timeout_s=300, folds=3)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")