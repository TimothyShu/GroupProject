import copy
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
import openml

import numpy as np
import torch

def preorder_traverse_xrfm_tree(node):
    """
    Preorder traversal for a single xRFM tree node dictionary.
    Visit order: node -> left subtree -> right subtree.
    Returns a flat list of node dicts (both split and leaf).
    """
    if node is None:
        return []

    nodes = [node]
    if node["type"] == "split":
        nodes.extend(preorder_traverse_xrfm_tree(node["left"]))
        nodes.extend(preorder_traverse_xrfm_tree(node["right"]))
    return nodes

def _retrain_xrfm(X_train, y_train, X_val, y_val, tuning_metric):
    # training ──────────────────────────────────────────────

    from hyperparameterTunning.xrfmparams import tunexrfm

    xrfmparams = tunexrfm(X_train, y_train, folds=3, trials=50)

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
            "get_agop_best_model": True,
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

    xrfm.fit(X_train.to_numpy(dtype=np.float32), y_train.to_numpy(), X_val.to_numpy(dtype=np.float32), y_val.to_numpy())

    return xrfm

if __name__ == "__main__":
    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))

    from normalizeFeatures import normalizeFeatures
    from utils import process_categorical_target, infer_task_and_metric
    from xrfmWithSubsetSize import xRFMWithSubsetSize as xRFM
    from PCA import calculatePCA
    from MI import calculateMI
    from permutationImportance import calculatePermutationImportance

    load_dotenv() 

    # we will use the diamonds dataset
    
    openml.config.apikey =  os.getenv("OPENML_KEY")
    # Load a binary classification dataset
    dataset = openml.datasets.get_dataset('diamonds')
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format='dataframe'
    )
    
    y = process_categorical_target(y)

    target_type, tuning_metric = infer_task_and_metric(y)

    # we will use the same procedure that we used for training and testing

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = normalizeFeatures(X_train, X_test)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    agop_path = Path(PROJECT_ROOT) / "xrfmModelAnalysis/diamonds_leaf_agops.pt"

    if agop_path.exists():
        best_agops = torch.load(agop_path, map_location="cuda", weights_only=False)
        print(f"Loaded {len(best_agops)} leaf AGOPs from {agop_path}")
    else:
        xrfm_model = _retrain_xrfm(X_train, y_train, X_val, y_val, tuning_metric)
        best_agops = xrfm_model.collect_best_agops()
        torch.save(best_agops, agop_path)
        print(f"Saved {len(best_agops)} leaf AGOPs to {agop_path}")
    
    xrfm_model = xRFM(device=torch.device('cuda'), tuning_metric="mse")
    state_dict = torch.load(f"{PROJECT_ROOT}/xrfmModelAnalysis/xrfm_model.pt", map_location="cuda", weights_only=False)
    xrfm_X_train_path = Path(PROJECT_ROOT) / "xrfmModelAnalysis/xrfm_X_train.npy"
    xrfm_X_train = torch.as_tensor(np.load(xrfm_X_train_path), dtype=torch.float32, device=torch.device('cuda'))
    xrfm_model.load_state_dict(state_dict, X_train=xrfm_X_train)

    #print("Leaf AGOPs:", best_agops)

    feature_names = X_train.columns.tolist()

    leaf_diagonals = [agop.diagonal() for agop in best_agops]
    leaf_avg_diagonal = sum(leaf_diagonals) / len(leaf_diagonals)

    print("\nAverage leaf AGOP diagonal (feature importance):")
    for name, val in sorted(zip(feature_names, leaf_avg_diagonal.tolist()), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {val:.6f}")

    _, pca = calculatePCA(X_train, len(feature_names))

    print("\nPCA top component loadings (feature importance):")
    top_loadings = pca.components_[0]  # first principal component could look at more
    for name, val in sorted(zip(feature_names, top_loadings.tolist()), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {name}: {val:.6f}")

    print("\nMI most correlated features (feature importance):")
    mi_series = calculateMI(X_train, y_train)
    for name, val in sorted(zip(feature_names, mi_series.tolist()), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {val:.6f}")

    print("\nPermutation importance (feature importance):")
    perm_importance = calculatePermutationImportance(xrfm_model, X_val, y_val)
    for name, val in sorted(zip(feature_names, perm_importance.tolist()), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {name}: {val:.6f}")