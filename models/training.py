import copy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier, TabPFNRegressor, save_fitted_tabpfn_model
import torch
from xgboost import XGBClassifier, XGBRegressor
from utils import infer_task_and_metric
from hyperparameterTunning.xrfmparams import tunexrfm
from hyperparameterTunning.xgboostparams import tunexgboost
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from xrfmWithSubsetSize import xRFMWithSubsetSize as xRFM

def train(X: pd.DataFrame, y: pd.Series, model_folder: str, refit: bool = False, hyperparameter_tuning_timeout_s: int | None = None, hyperparameter_tuning_folds: int = 3, tabpfn_context_sizes: list[int] | None = None, trials: int | None = None):
    """This is an example of a training function that will train the 3 models on the same data and save the model for later testing
    Args:
        X (pd.DataFrame): Training features
        y (pd.Series): Training targets
        model_folder (str): Folder where the trained model will be saved
        tabpfn_context_sizes (list[int] | None): List of context sizes to train TabPFN with. Defaults to [500, 1000, 2000].

    """

    mopdel_path = Path(model_folder)
    mopdel_path.mkdir(parents=True, exist_ok=True)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    _, tuning_metric = infer_task_and_metric(y)

    xrfm_tuning_time = None
    xrfm_fit_time = None
    xgboost_tuning_time = None
    xgboost_fit_time = None
    tabpfn_times = {}  # context_size -> fit_time

    if tabpfn_context_sizes is None:
        tabpfn_context_sizes = [500, 1000, 2000]
    
    # train only when model does not exist or refit is True
    if not (mopdel_path / "xrfm_model.pt").exists() or refit:
        tuning_t, fit_t, n_trials = _train_xrfm(X_train, y_train, X_val, y_val, hyperparameter_tuning_timeout_s, hyperparameter_tuning_folds, tuning_metric, model_folder, trials)
        xrfm_tuning_time = tuning_t
        xrfm_fit_time = fit_t
        xrfm_tunning_trials = n_trials
    
    if not (mopdel_path / "xgboost_model.json").exists() or refit:
        tuning_t, fit_t, n_trials = _train_xgboost(X_train, y_train, X_val, y_val, hyperparameter_tuning_timeout_s, hyperparameter_tuning_folds, tuning_metric, model_folder, trials)
        xgboost_tuning_time = tuning_t
        xgboost_fit_time = fit_t
        xgboost_fit_trials = n_trials
    
    for ctx_size in tabpfn_context_sizes:
        model_name = f"tabpfn_model_ctx{ctx_size}.tabpfn_fit"
        if not (mopdel_path / model_name).exists() or refit:
            start = time.perf_counter()
            _train_tabpfn(X_train, y_train, model_folder, context_size=ctx_size)
            tabpfn_times[ctx_size] = time.perf_counter() - start
    
    print("\nTraining times----------------------------------")
    if xrfm_tuning_time is not None:
        print(f"xRFM Tuning Time: {xrfm_tuning_time:.2f}s | Fit Time: {xrfm_fit_time:.2f}s | Total: {xrfm_tuning_time + xrfm_fit_time:.2f}s")
        print(f"xRFM Fit Time per Sample: {xrfm_fit_time / len(X_train):.6f} seconds")
        print(f"xRFM Tuning Trials: {xrfm_tunning_trials}")
    if xgboost_tuning_time is not None:
        print(f"XGBoost Tuning Time: {xgboost_tuning_time:.2f}s | Fit Time: {xgboost_fit_time:.2f}s | Total: {xgboost_tuning_time + xgboost_fit_time:.2f}s")
        print(f"XGBoost Fit Time per Sample: {xgboost_fit_time / len(X_train):.6f} seconds")
        print(f"XGBoost Tuning Trials: {xgboost_fit_trials}")
    for ctx_size, t in tabpfn_times.items():
        print(f"TabPFN (ctx={ctx_size}) Fit Time: {t:.2f}s (no tuning) | per Sample: {t / min(ctx_size, len(X_train)):.6f}s")

def _train_xrfm(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, timeout_s: int | None, folds: int, tuning_metric: str, model_folder: str, n_trials: int) -> tuple[float, float, int]:
    """This is a helper function to train xRFM with hyperparameter tuning, we separate it out from the main train function to make it easier to call from the hyperparameter tuning function without having to run the whole training process
    Args:
        X_train (pd.DataFrame): Training features, needed xrfm to set centers
        y_train (pd.Series): Training targets, needed for tuning
        X_val (pd.DataFrame): Validation features, needed for tuning
        y_val (pd.Series): Validation targets, needed for tuning
        n_trials (int): Number of trials for hyperparameter tuning
        timeout_s (int | None): Time limit for training in seconds
        folds (int): Number of folds for cross-validation during tuning
        tuning_metric (str): The metric to optimize during tuning, either "mse" for regression or "accuracy" for classification
        model_folder (str): Folder where the trained model is saved
    """

    # xRFM
    tune_start = time.perf_counter()
    xrfmparams = tunexrfm(X_train, y_train, timeout_s=timeout_s, folds=folds, trials=n_trials)
    tuning_time = time.perf_counter() - tune_start
    n_trials = len(xrfmparams.trials)

    best_xrfm_params = xrfmparams.best_params

    leaf_rfm_params = {
        "model": {
            "kernel": "lpq",
            "bandwidth": np.log(best_xrfm_params["bandwidth"]),
            "norm_p": best_xrfm_params["norm_p"],
            "exponent": best_xrfm_params["exponent"],
        },
        "fit": {
            "reg": best_xrfm_params["reg"],
            "iters": 0,
            "return_best_params": True,
            "verbose": False,
        },
    }

    split_rfm_params = {
        "model": copy.deepcopy(leaf_rfm_params["model"]),
        "fit": {
            "get_agop_best_model": True,
            "return_best_params": False,
            "reg": best_xrfm_params["reg"],
            "iters": 0,
            "early_stop_rfm": False,
            "verbose": False,
        },
    }

    xrfm_params = {
        # min subset size as a proportion of the data (to prevent overfitting and ensure enough samples in each leaf)
        "max_leaf_size": int(best_xrfm_params["subset_prop"] * len(X_train)),
        "use_temperature_tuning": True, # we turned this off for validation to speed it up, but we will turn it on for the final training
        "rfm_params": leaf_rfm_params,
        "default_rfm_params": split_rfm_params,
    }

    xrfm_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split_subset_size = max(1000, int(0.8 * xrfm_params["max_leaf_size"]))
    xrfm = xRFM(**xrfm_params, device=xrfm_device, tuning_metric=tuning_metric, split_subset_size=split_subset_size)

    # Training

    fit_start = time.perf_counter()
    try:
        xrfm.fit(X_train.to_numpy(dtype=np.float32), y_train.to_numpy(), X_val.to_numpy(dtype=np.float32), y_val.to_numpy())
    except RuntimeError as error:
        if 'Boolean value of Tensor with more than one value is ambiguous' in str(error):
            print('xRFM CUDA path failed; falling back to CPU for final fit.')
            xrfm = xRFM(**xrfm_params, device=torch.device('cpu'), tuning_metric=tuning_metric)
            xrfm.fit(X_train.to_numpy(dtype=np.float32), y_train.to_numpy(), X_val.to_numpy(dtype=np.float32), y_val.to_numpy())
        else:
            raise
    fit_time = time.perf_counter() - fit_start

    # save
    torch.save(xrfm.get_state_dict(), f"{model_folder}/xrfm_model.pt")
    np.save(f"{model_folder}/xrfm_X_train.npy", X_train.to_numpy(dtype=np.float32)) # needed for reconstruction

    return tuning_time, fit_time, n_trials

def _train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, timeout_s: int | None, folds: int, tuning_metric: str, model_folder: str, n_trials: int) -> tuple[float, float, int]:
    """This is a helper function to train xgboost with hyperparameter tuning, we separate it out from the main train function to make it easier to call from the hyperparameter tuning function without having to run the whole training process
    Args:
        X_train (pd.DataFrame): Training features, needed xrfm to set centers
        y_train (pd.Series): Training targets, needed for tuning
        X_val (pd.DataFrame): Validation features, needed for tuning
        y_val (pd.Series): Validation targets, needed for tuning
        timeout_s (int | None): Time limit for training in seconds
        folds (int): Number of folds for cross-validation during tuning
        tuning_metric (str): The metric to optimize during tuning, either "mse" for regression or "accuracy" for classification
        model_folder (str): Folder where the trained model is saved
    """

    # xgboost
    tune_start = time.perf_counter()
    xgboostparams = tunexgboost(X_train, y_train, timeout_s=timeout_s, folds=folds, trials=n_trials)
    tuning_time = time.perf_counter() - tune_start
    n_trials = len(xgboostparams.trials)

    xgb_n_estimators = 2000
    early_stopping_rounds = 50
    best_xgboost_params = xgboostparams.best_params
    xgb_device = "cuda" if torch.cuda.is_available() else "cpu"
    best_xgboost_params = {
        "eta": best_xgboost_params["eta"],
        "n_estimators": xgb_n_estimators,
        "gamma": best_xgboost_params["gamma"],
        "max_depth": best_xgboost_params["max_depth"],
        "min_child_weight": best_xgboost_params["min_child_weight"],
        "subsample": best_xgboost_params["subsample"],
        "colsample_bytree": best_xgboost_params["colsample_bytree"],
        "reg_alpha": best_xgboost_params["reg_alpha"],
        "reg_lambda": best_xgboost_params["reg_lambda"],
        "tree_method": "hist",
        "early_stopping_rounds": early_stopping_rounds,
        "device": xgb_device,
        "random_state": 42,
    }

    if tuning_metric == "mse":
        xgboost = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            **best_xgboost_params,
        )
    else:
        n_classes = int(np.max(y_train.to_numpy())) + 1
        if n_classes <= 2:
            xgboost = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                **best_xgboost_params,
            )
        else:
            xgboost = XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                num_class=n_classes,
                **best_xgboost_params,
            )

    # training
    X_train = X_train.to_numpy(dtype=np.float32)
    X_val = X_val.to_numpy(dtype=np.float32)
    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    fit_start = time.perf_counter()
    xgboost.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    fit_time = time.perf_counter() - fit_start

    # save
    xgboost.save_model(f"{model_folder}/xgboost_model.json")

    return tuning_time, fit_time, n_trials

def _train_tabpfn(X_train: pd.DataFrame, y_train: pd.Series, model_folder: str, context_size: int = 1000):
    """This is a helper function to train tabPFN, we separate it out from the main train function to make it easier to call from the hyperparameter tuning function without having to run the whole training process
    Args:
        X_train (pd.DataFrame): Training features, needed xrfm to set centers
        y_train (pd.Series): Training targets, needed for tuning
        model_folder (str): Folder where the trained model is saved
        context_size (int): The number of training samples to use as context for tabPFN, default is 1000, anything over 2-4000 is very slow so be careful increasing this
    """

    task_type, _ = infer_task_and_metric(y_train)

    if task_type == "categorical":
        tabPFN = TabPFNClassifier(device="cuda" if torch.cuda.is_available() else "cpu")
    else:
        tabPFN = TabPFNRegressor(device="cuda" if torch.cuda.is_available() else "cpu")

    # limit size of training data to reduce inference time

    if len(X_train) > context_size:
        X_train = X_train.iloc[:context_size]
        y_train = y_train.iloc[:context_size]

    tabPFN.fit(X_train.to_numpy(), y_train.to_numpy())

    save_fitted_tabpfn_model(tabPFN, f"{model_folder}/tabpfn_model_ctx{context_size}.tabpfn_fit")