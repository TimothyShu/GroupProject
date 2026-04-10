import numpy as np
import pandas as pd
from pathlib import Path
from tabpfn import load_fitted_tabpfn_model
import torch
from xrfm import xRFM
import xgboost as xgb
import time

from utils import infer_task_and_metric

def _get_auc_roc(y_true, y_pred):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_pred)

def test(X: pd.DataFrame, y: pd.Series, model_folder: str):
    """This is an example of a testing function that will load the saved models from training and evaluate them on the test set
    Args:
        X (pd.DataFrame): Test features
        y (pd.Series): Test targets
        model_folder (str): Folder where the trained model is saved
    """

    # now load the models and test them

    # xRFM
    print("Loading xRFM model...")
    xrfm_model = xRFM(device=torch.device('cpu'))
    state_dict = torch.load(f"{model_folder}/xrfm_model.pt", map_location="cpu", weights_only=False)
    # load_state_dict uses train_indices stored in each leaf node to index into this
    # matrix to set kernel centers — it must be the exact same array used at fit time
    xrfm_X_train_path = Path(model_folder) / "xrfm_X_train.npy"
    if xrfm_X_train_path.exists():
        xrfm_X_train = torch.as_tensor(np.load(xrfm_X_train_path), dtype=torch.float32)
    else:
        raise FileNotFoundError(
            f"{xrfm_X_train_path} not found. Re-run training so the exact fit "
            "matrix is saved alongside the model."
        )
    xrfm_model.load_state_dict(
        state_dict,
        X_train=xrfm_X_train,
    )
    print("xRFM model loaded.")
    
    # XGBoost
    print("Loading XGBoost model...")
    if infer_task_and_metric(y)[0] == "categorical":
        xgb_model = xgb.XGBClassifier()
    else:
        xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(f"{model_folder}/xgboost_model.json")
    print("XGBoost model loaded.")

    # tabPFN
    print("Loading TabPFN model...")
    tabpfn_model = load_fitted_tabpfn_model(f"{model_folder}/tabpfn_model.tabpfn_fit", device="cuda" if torch.cuda.is_available() else "cpu")
    print("TabPFN model loaded.")

    # Evaluate the models on the test set

    print("Evaluating models on the test set...")
    print("Evaluating xRFM...")
    start = time.perf_counter()
    xrfm_preds = xrfm_model.predict(X.to_numpy(dtype=np.float32))
    end = time.perf_counter()
    xrfm_duration = end - start
    xrfm_time_per_sample = xrfm_duration / len(X)
    print("Evaluating XGBoost...")
    start = time.perf_counter()
    xgb_preds = xgb_model.predict(X)
    end = time.perf_counter()
    xgb_duration = end - start
    xgb_time_per_sample = xgb_duration / len(X)
    print("Evaluating TabPFN...")
    start = time.perf_counter()
    tabpfn_preds = tabpfn_model.predict(X.to_numpy())
    end = time.perf_counter()
    tabpfn_duration = end - start
    tabpfn_time_per_sample = tabpfn_duration / len(X)
    print("Evaluation complete.")

    # Here you can add code to calculate metrics like MSE or accuracy depending on the task and print them out
    task_type, _ = infer_task_and_metric(y)

    if task_type == "regression":
        xrfm_mse = np.mean((xrfm_preds.reshape(-1) - y.to_numpy().reshape(-1)) ** 2)
        xgb_mse = np.mean((xgb_preds.reshape(-1) - y.to_numpy().reshape(-1)) ** 2)
        tabpfn_mse = np.mean((tabpfn_preds.reshape(-1) - y.to_numpy().reshape(-1)) ** 2)

        print("\nxRFM performance----------------------------------")
        print(f"xRFM MSE: {xrfm_mse:.4f}")
        print(f"xRFM Inference Time per Sample: {xrfm_time_per_sample:.6f} seconds")
        print("\nXGBoost performance-------------------------------")
        print(f"XGBoost MSE: {xgb_mse:.4f}")
        print(f"XGBoost Inference Time per Sample: {xgb_time_per_sample:.6f} seconds")
        print("\nTabPFN performance--------------------------------")
        print(f"TabPFN MSE: {tabpfn_mse:.4f}")
        print(f"TabPFN Inference Time per Sample: {tabpfn_time_per_sample:.6f} seconds")
    else:
        xrfm_acc = np.mean(np.asarray(xrfm_preds).reshape(-1) == y.to_numpy().reshape(-1))
        xrfm_auc_roc = _get_auc_roc(y.to_numpy().reshape(-1), np.asarray(xrfm_preds).reshape(-1))
        xgb_acc = np.mean(np.asarray(xgb_preds).reshape(-1) == y.to_numpy().reshape(-1))
        xgb_auc_roc = _get_auc_roc(y.to_numpy().reshape(-1), np.asarray(xgb_preds).reshape(-1))
        tabpfn_acc = np.mean(np.asarray(tabpfn_preds).reshape(-1) == y.to_numpy().reshape(-1))
        tabpfn_auc_roc = _get_auc_roc(y.to_numpy().reshape(-1), np.asarray(tabpfn_preds).reshape(-1))

        print("\nxRFM performance----------------------------------")
        print(f"xRFM Accuracy: {xrfm_acc:.4f}")
        print(f"xRFM AUC-ROC: {xrfm_auc_roc:.4f}")
        print(f"xRFM Inference Time per Sample: {xrfm_time_per_sample:.6f} seconds")
        print("\nXGBoost performance-------------------------------")
        print(f"XGBoost Accuracy: {xgb_acc:.4f}")
        print(f"XGBoost AUC-ROC: {xgb_auc_roc:.4f}")
        print(f"XGBoost Inference Time per Sample: {xgb_time_per_sample:.6f} seconds")
        print("\nTabPFN performance--------------------------------")
        print(f"TabPFN Accuracy: {tabpfn_acc:.4f}")
        print(f"TabPFN AUC-ROC: {tabpfn_auc_roc:.4f}")
        print(f"TabPFN Inference Time per Sample: {tabpfn_time_per_sample:.6f} seconds")