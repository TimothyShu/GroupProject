
import copy

import numpy as np
import optuna
from optuna.samplers import TPESampler
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
import torch
from xrfm import xRFM
from resxrfm import xRFM_res

from utils import infer_task_and_metric

subset_prop = 0.5 # use this so that all splits are same size for comparison

def train(X: pd.DataFrame, y: pd.Series, hyperparameter_tuning_timeout_s: int | None = None, hyperparameter_tuning_folds: int = 3, trials: int | None = None, same_splits: bool = True) -> tuple[xRFM, xRFM_res]:
    """This is an example of a training function that will train the 3 models on the same data and save the model for later testing
    Args:
        X (pd.DataFrame): Training features
        y (pd.Series): Training targets
        tabpfn_context_sizes (list[int] | None): List of context sizes to train TabPFN with. Defaults to [500, 1000, 2000].

    """

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    _, tuning_metric = infer_task_and_metric(y)
    
    xrfm = _train_xrfm(X_train, y_train, X_val, y_val, hyperparameter_tuning_timeout_s, hyperparameter_tuning_folds, tuning_metric, trials, same_splits)

    xrfm_res = _train_resxrfm(X_train, y_train, X_val, y_val, hyperparameter_tuning_timeout_s, hyperparameter_tuning_folds, tuning_metric, trials, same_splits)

    return xrfm, xrfm_res

def _train_xrfm(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, timeout_s: int | None, folds: int, tuning_metric: str, n_trials: int, same_splits: bool) -> xRFM:
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
    """

    # xRFM
    xrfmparams = tunexrfm(X_train, y_train, timeout_s=timeout_s, folds=folds, trials=n_trials, same_splits=same_splits)

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
        "max_leaf_size": int(subset_prop * len(X_train)) if same_splits else best_xrfm_params["max_leaf_size"],
        "use_temperature_tuning": True, # we turned this off for validation to speed it up, but we will turn it on for the final training
        "rfm_params": leaf_rfm_params,
        "default_rfm_params": split_rfm_params,
    }

    xrfm_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xrfm = xRFM(**xrfm_params, device=xrfm_device, tuning_metric=tuning_metric)

    # Training

    xrfm.fit(X_train.to_numpy(dtype=np.float32), y_train.to_numpy(), X_val.to_numpy(dtype=np.float32), y_val.to_numpy())

    return xrfm

def _train_resxrfm(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, timeout_s: int | None, folds: int, tuning_metric: str, n_trials: int, same_splits: bool) -> xRFM_res:
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
    """

    # xRFM
    xrfmparams = tuneresxrfm(X_train, y_train, timeout_s=timeout_s, folds=folds, trials=n_trials, same_splits=same_splits)

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
        "max_leaf_size": int(subset_prop * len(X_train)) if same_splits else best_xrfm_params["max_leaf_size"],
        "use_temperature_tuning": True, # we turned this off for validation to speed it up, but we will turn it on for the final training
        "rfm_params": leaf_rfm_params,
        "default_rfm_params": split_rfm_params,
    }

    xrfm_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xrfm = xRFM_res(**xrfm_params, device=xrfm_device, tuning_metric=tuning_metric)

    # Training

    xrfm.fit(X_train.to_numpy(dtype=np.float32), y_train.to_numpy(), X_val.to_numpy(dtype=np.float32), y_val.to_numpy())

    return xrfm

def _objective(trial: optuna.trial.Trial, X: pd.DataFrame, y: pd.Series, time_limit_s: int | None = None, folds: int = 5, same_splits: bool = False):

    preferred_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task_type, tuning_metric = infer_task_and_metric(y)

    # Search space for xRFM-specific hyperparameters

    bandwidth = trial.suggest_float("bandwidth", 1, 200, log=True)
    exponent =  trial.suggest_float("exponent", 0.7, 1.4)
    norm_p = trial.suggest_float("norm_p", exponent, exponent + 0.8*(2-exponent))
    reg = trial.suggest_float("reg", 1e-6, 10, log=True)

    if not same_splits:
        max_leaf_size = trial.suggest_int("max_leaf_size", int(0.01 * len(X)), int(0.5 * len(X)))
    else:
        max_leaf_size = int(subset_prop * len(X))

    rfm_params = {
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
            "verbose": False,
        },
    }

    default_rfm_params = {
        "model": copy.deepcopy(rfm_params["model"]),
        "fit": {
            "get_agop_best_model": True,
            "return_best_params": False,
            "reg": reg,
            "iters": 0,
            "early_stop_rfm": False,
            "verbose": False,
        },
    }

    if time_limit_s is not None:
        params = {
            # min subset size as a proportion of the data (to prevent overfitting and ensure enough samples in each leaf)
            "max_leaf_size": max_leaf_size,
            "use_temperature_tuning": False, # to speed up validation, but will be turned on for the final training
            "time_limit_s": time_limit_s,
            "rfm_params": rfm_params,
            "default_rfm_params": default_rfm_params,
        }
    else:
        params = {
            "max_leaf_size": max_leaf_size,
            "use_temperature_tuning": False,
            "rfm_params": rfm_params,
            "default_rfm_params": default_rfm_params,
        }

    # run kfold cross validation on the training data
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    result = 0
    for step, (train_index, val_index) in enumerate(kf.split(X)):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        X_train_arr = X_train_fold.to_numpy(dtype=np.float32)
        X_val_arr = X_val_fold.to_numpy(dtype=np.float32)
        y_train_arr = y_train_fold.to_numpy()
        y_val_arr = y_val_fold.to_numpy()

        # Initialize and train model (CUDA first, fallback to CPU on known xRFM CUDA bug)
        model = xRFM(**params, device=preferred_device, tuning_metric=tuning_metric)
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

def tunexrfm(X: pd.DataFrame, y: pd.Series, timeout_iteration: int | None = None, timeout_s: int | None = None, folds: int = 5, trials: int | None = None, same_splits: bool = False) -> optuna.Study:
    """Tune xRFM hyperparameters using Optuna

    Args:
        X (pd.DataFrame): Training features
        y (pd.Series): Training targets
        timeout_iteration (int | None, optional): Time limit for optimization in iterations. Defaults to None.
        timeout_s (int | None, optional): Time limit for optimization in seconds. Defaults to None (no time limit).
        folds (int, optional): Number of folds for cross-validation. Defaults to 5.
        trials (int | None, optional): Number of trials for hyperparameter tuning. Defaults to None (no limit).
    Returns:
        optuna.Study: The Optuna study object containing the results of the optimization
    """
    
    _, tuning_metric = infer_task_and_metric(y)
    direction = "minimize" if tuning_metric == "mse" else "maximize"

    sampler = TPESampler(seed=42, multivariate=True)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1)

    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    try:
        if trials is not None:
            study.optimize(lambda trial: _objective(trial, X, y, timeout_iteration, folds, same_splits), n_trials=trials)
        else:
            study.optimize(lambda trial: _objective(trial, X, y, folds, same_splits), timeout=timeout_s)
    except KeyboardInterrupt:
        print("Optimization interrupted. Returning completed trials so far.")
    return study

def _resobjective(trial: optuna.trial.Trial, X: pd.DataFrame, y: pd.Series, time_limit_s: int | None = None, folds: int = 5, same_splits: bool = False):

    preferred_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task_type, tuning_metric = infer_task_and_metric(y)

    # Search space for xRFM-specific hyperparameters

    bandwidth = trial.suggest_float("bandwidth", 1, 200, log=True)
    exponent =  trial.suggest_float("exponent", 0.7, 1.4)
    norm_p = trial.suggest_float("norm_p", exponent, exponent + 0.8*(2-exponent))
    reg = trial.suggest_float("reg", 1e-6, 10, log=True)
    
    if not same_splits:
        max_leaf_size = trial.suggest_int("max_leaf_size", int(0.01 * len(X)), int(0.5 * len(X)))
    else:
        max_leaf_size = int(subset_prop * len(X))

    rfm_params = {
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
            "verbose": False,
        },
    }

    default_rfm_params = {
        "model": copy.deepcopy(rfm_params["model"]),
        "fit": {
            "get_agop_best_model": True,
            "return_best_params": False,
            "reg": reg,
            "iters": 0,
            "early_stop_rfm": False,
            "verbose": False,
        },
    }

    if time_limit_s is not None:
        params = {
            # min subset size as a proportion of the data (to prevent overfitting and ensure enough samples in each leaf)
            "max_leaf_size": max_leaf_size,
            "use_temperature_tuning": False, # to speed up validation, but will be turned on for the final training
            "time_limit_s": time_limit_s,
            "rfm_params": rfm_params,
            "default_rfm_params": default_rfm_params,
        }
    else:
        params = {
            "max_leaf_size": max_leaf_size,
            "use_temperature_tuning": False,
            "rfm_params": rfm_params,
            "default_rfm_params": default_rfm_params,
        }

    # run kfold cross validation on the training data
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    result = 0
    for step, (train_index, val_index) in enumerate(kf.split(X)):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        X_train_arr = X_train_fold.to_numpy(dtype=np.float32)
        X_val_arr = X_val_fold.to_numpy(dtype=np.float32)
        y_train_arr = y_train_fold.to_numpy()
        y_val_arr = y_val_fold.to_numpy()

        # Initialize and train model (CUDA first, fallback to CPU on known xRFM CUDA bug)
        model = xRFM_res(**params, device=preferred_device, tuning_metric=tuning_metric)
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

def tuneresxrfm(X: pd.DataFrame, y: pd.Series, timeout_iteration: int | None = None, timeout_s: int | None = None, folds: int = 5, trials: int | None = None, same_splits: bool = False) -> optuna.Study:
    """Tune xRFM hyperparameters using Optuna

    Args:
        X (pd.DataFrame): Training features
        y (pd.Series): Training targets
        timeout_iteration (int | None, optional): Time limit for optimization in iterations. Defaults to None.
        timeout_s (int | None, optional): Time limit for optimization in seconds. Defaults to None (no time limit).
        folds (int, optional): Number of folds for cross-validation. Defaults to 5.
        trials (int | None, optional): Number of trials for hyperparameter tuning. Defaults to None (no limit).
    Returns:
        optuna.Study: The Optuna study object containing the results of the optimization
    """
    
    _, tuning_metric = infer_task_and_metric(y)
    direction = "minimize" if tuning_metric == "mse" else "maximize"

    sampler = TPESampler(seed=42, multivariate=True)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1)

    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    try:
        if trials is not None:
            study.optimize(lambda trial: _resobjective(trial, X, y, timeout_iteration, folds, same_splits), n_trials=trials)
        else:
            study.optimize(lambda trial: _resobjective(trial, X, y, folds, same_splits), timeout=timeout_s)
    except KeyboardInterrupt:
        print("Optimization interrupted. Returning completed trials so far.")
    return study