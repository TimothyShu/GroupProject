import copy
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier, TabPFNRegressor
import torch
from xgboost import XGBClassifier, XGBRegressor
from xrfm import xRFM
from hyperparameterTunning.utils import infer_task_and_metric
from hyperparameterTunning.xrfmparams import tunexrfm
from hyperparameterTunning.xgboostparams import tunexgboost


def train(X: pd.DataFrame, y: pd.Series):
    """This is an example of a training function that will train the 3 models on the same data and save the model for later testing
    Args:
        X (pd.DataFrame): Training features
        y (pd.Series): Training targets

    """

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    _, tuning_metric = infer_task_and_metric(y)


    # keep these the same for all models for a fair comparison
    n_trials = 50
    timeout_s = 3600
    folds = 5

    # Hyper parameter tunning

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

    # xgboost
    xgboostparams = tunexgboost(X_val, y_val, n_trials=n_trials, timeout_s=timeout_s, folds=folds)

    # xgboostparams earch is relatively fast so we did not use any params to speed it up, we can just use default params for the final training

    best_xgboost_params = xgboostparams.best_params

    if tuning_metric == "mse":
        xgboost = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            **best_xgboost_params,
        )
    else:
        n_classes = int(np.max(y.to_numpy())) + 1
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

    # no need for tabPFN since no hyperparameters to tune

    if tuning_metric == "mse":
        tabPFN = TabPFNRegressor(device="cuda" if torch.cuda.is_available() else "cpu")
    else:
        tabPFN = TabPFNClassifier(device="cuda" if torch.cuda.is_available() else "cpu")

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

    xgboost.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

    tabPFN.fit(X_train.to_numpy(), y_train.to_numpy())

    # save all the models for later testing
    torch.save(xrfm.get_state_dict(), "xrfm_model.pt")
    xgboost.save_model("xgboost_model.json")
    
    with open('tabpfn_model.pkl', 'wb') as f:
        pickle.dump(tabPFN, f)