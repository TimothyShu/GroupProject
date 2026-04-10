import numpy as np
import pandas as pd
from pathlib import Path
from tabpfn import load_fitted_tabpfn_model
import torch
from xrfm import xRFM
import xgboost as xgb

from utils import infer_task_and_metric


def test(X: pd.DataFrame, y: pd.Series, X_train: pd.DataFrame, model_folder: str):
    """This is an example of a testing function that will load the saved models from training and evaluate them on the test set
    Args:
        X (pd.DataFrame): Test features
        y (pd.Series): Test targets
        X_train (pd.DataFrame): Training features, needed xrfm to set centers
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
    xrfm_preds = xrfm_model.predict(X.to_numpy(dtype=np.float32))
    print("Evaluating XGBoost...")
    xgb_preds = xgb_model.predict(X)
    print("Evaluating TabPFN...")
    tabpfn_preds = tabpfn_model.predict(X.to_numpy())
    print("Evaluation complete.")

    # Here you can add code to calculate metrics like MSE or accuracy depending on the task and print them out
    task_type, _ = infer_task_and_metric(y)

    if task_type == "regression":
        xrfm_mse = np.mean((xrfm_preds.reshape(-1) - y.to_numpy().reshape(-1)) ** 2)
        xgb_mse = np.mean((xgb_preds.reshape(-1) - y.to_numpy().reshape(-1)) ** 2)
        tabpfn_mse = np.mean((tabpfn_preds.reshape(-1) - y.to_numpy().reshape(-1)) ** 2)

        print(f"xRFM Test MSE: {xrfm_mse:.4f}")
        print(f"XGBoost Test MSE: {xgb_mse:.4f}")
        print(f"TabPFN Test MSE: {tabpfn_mse:.4f}")
    else:
        xrfm_acc = np.mean(np.asarray(xrfm_preds).reshape(-1) == y.to_numpy().reshape(-1))
        xgb_acc = np.mean(np.asarray(xgb_preds).reshape(-1) == y.to_numpy().reshape(-1))
        tabpfn_acc = np.mean(np.asarray(tabpfn_preds).reshape(-1) == y.to_numpy().reshape(-1))

        print(f"xRFM Test Accuracy: {xrfm_acc:.4f}")
        print(f"XGBoost Test Accuracy: {xgb_acc:.4f}")
        print(f"TabPFN Test Accuracy: {tabpfn_acc:.4f}")