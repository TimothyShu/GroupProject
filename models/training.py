import copy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier, TabPFNRegressor, save_fitted_tabpfn_model
import torch
from xgboost import XGBClassifier, XGBRegressor
from xrfm import xRFM
from utils import infer_task_and_metric
from hyperparameterTunning.xrfmparams import tunexrfm
from hyperparameterTunning.xgboostparams import tunexgboost
import time


def train(X: pd.DataFrame, y: pd.Series, model_folder: str):
    """This is an example of a training function that will train the 3 models on the same data and save the model for later testing
    Args:
        X (pd.DataFrame): Training features
        y (pd.Series): Training targets
        model_folder (str): Folder where the trained model will be saved

    """

    mopdel_path = Path(model_folder)
    mopdel_path.mkdir(parents=True, exist_ok=True)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    _, tuning_metric = infer_task_and_metric(y)


    # keep these the same for all models for a fair comparison
    n_trials = 50
    timeout_s = 3600
    folds = 5

    xrfm_training_time = None
    xgboost_training_time = None
    tabpfn_training_time = None
    
    # train only when model does not exist
    if not (mopdel_path / "xrfm_model.pt").exists():
        start = time.perf_counter()
        _train_xrfm(X_train, y_train, X_val, y_val, n_trials, timeout_s, folds, tuning_metric, model_folder)
        end = time.perf_counter()
        xrfm_training_time = end - start
        xrfm_training_time_per_sample = xrfm_training_time / len(X_train)
    
    if not (mopdel_path / "xgboost_model.json").exists():
        start = time.perf_counter()
        _train_xgboost(X_train, y_train, X_val, y_val, n_trials, timeout_s, folds, tuning_metric, model_folder)
        end = time.perf_counter()
        xgboost_training_time = end - start
        xgboost_training_time_per_sample = xgboost_training_time / len(X_train)
    
    if not (mopdel_path / "tabpfn_model.tabpfn_fit").exists():
        start = time.perf_counter()
        _train_tabpfn(X_train, y_train, model_folder)
        end = time.perf_counter()
        tabpfn_training_time = end - start
        tabpfn_training_time_per_sample = tabpfn_training_time / len(X_train)
    
    print("\nTraining times----------------------------------")
    if xrfm_training_time is not None:
        print(f"xRFM Training Time per Sample: {xrfm_training_time_per_sample:.6f} seconds")
    if xgboost_training_time is not None:
        print(f"XGBoost Training Time per Sample: {xgboost_training_time_per_sample:.6f} seconds")
    if tabpfn_training_time is not None:
        print(f"TabPFN Training Time per Sample: {tabpfn_training_time_per_sample:.6f} seconds")

def _train_xrfm(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, n_trials: int, timeout_s: int, folds: int, tuning_metric: str, model_folder: str):
    """This is a helper function to train xRFM with hyperparameter tuning, we separate it out from the main train function to make it easier to call from the hyperparameter tuning function without having to run the whole training process
    Args:
        X_train (pd.DataFrame): Training features, needed xrfm to set centers
        y_train (pd.Series): Training targets, needed for tuning
        X_val (pd.DataFrame): Validation features, needed for tuning
        y_val (pd.Series): Validation targets, needed for tuning
        n_trials (int): Number of trials for hyperparameter tuning
        timeout_s (int): Time limit for training in seconds
        folds (int): Number of folds for cross-validation during tuning
        tuning_metric (str): The metric to optimize during tuning, either "mse" for regression or "accuracy" for classification
        model_folder (str): Folder where the trained model is saved
    """

    # xRFM
    timeout_iteration = 5 # might want to increase this for better results, but it will take longer to run
    xrfmparams = tunexrfm(X_val, y_val, n_trials=n_trials, timeout_iteration=timeout_iteration, timeout_s=timeout_s, folds=folds)

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
    xrfm = xRFM(**xrfm_params, device=xrfm_device, tuning_metric=tuning_metric)

    # Training

    try:
        xrfm.fit(X_train.to_numpy(dtype=np.float32), y_train.to_numpy(), X_val.to_numpy(dtype=np.float32), y_val.to_numpy())
    except RuntimeError as error:
        if xrfm_device.type == 'cuda' and 'Boolean value of Tensor with more than one value is ambiguous' in str(error):
            print('xRFM CUDA path failed; falling back to CPU for final fit.')
            xrfm = xRFM(**xrfm_params, device=torch.device('cpu'), tuning_metric=tuning_metric)
            xrfm.fit(X_train.to_numpy(dtype=np.float32), y_train.to_numpy(), X_val.to_numpy(dtype=np.float32), y_val.to_numpy())
        else:
            raise

    # save
    torch.save(xrfm.get_state_dict(), f"{model_folder}/xrfm_model.pt")
    np.save(f"{model_folder}/xrfm_X_train.npy", X_train.to_numpy(dtype=np.float32)) # needed for reconstruction

def _train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, n_trials: int, timeout_s: int, folds: int, tuning_metric: str, model_folder: str) -> XGBClassifier | XGBRegressor:
    """This is a helper function to train xgboost with hyperparameter tuning, we separate it out from the main train function to make it easier to call from the hyperparameter tuning function without having to run the whole training process
    Args:
        X_train (pd.DataFrame): Training features, needed xrfm to set centers
        y_train (pd.Series): Training targets, needed for tuning
        X_val (pd.DataFrame): Validation features, needed for tuning
        y_val (pd.Series): Validation targets, needed for tuning
        n_trials (int): Number of trials for hyperparameter tuning
        timeout_s (int): Time limit for training in seconds
        folds (int): Number of folds for cross-validation during tuning
        tuning_metric (str): The metric to optimize during tuning, either "mse" for regression or "accuracy" for classification
        model_folder (str): Folder where the trained model is saved
    """

    # xgboost
    xgboostparams = tunexgboost(X_val, y_val, n_trials=n_trials, timeout_s=timeout_s, folds=folds)

    best_xgboost_params = xgboostparams.best_params

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
    xgboost.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

    # save
    xgboost.save_model(f"{model_folder}/xgboost_model.json")

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

    save_fitted_tabpfn_model(tabPFN, f"{model_folder}/tabpfn_model.tabpfn_fit")