import pandas as pd
import torch
from xrfm import xRFM


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
    xrfm_model = xRFM(device=torch.device('cpu'))
    state_dict = torch.load(f"{model_folder}/xrfm_model.pt", map_location="cpu", weights_only=False)
    xrfm_model.load_state_dict(
        state_dict,
        X_train=torch.as_tensor(X_train.to_numpy(copy=True), dtype=torch.float32),
    )
    
    # XGBoost

    