import numpy as np
import pandas as pd
from pathlib import Path
from tabpfn import load_fitted_tabpfn_model
import torch
from xrfm import xRFM
import xgboost as xgb
import time
import re

from utils import infer_task_and_metric


def _safe_auc(y_true, proba):
    """Compute AUC-ROC from probability scores, handling binary and multiclass."""
    from sklearn.metrics import roc_auc_score
    if proba is None:
        return float('nan')
    proba = np.asarray(proba)
    if proba.ndim == 2 and proba.shape[1] == 2:
        return roc_auc_score(y_true, proba[:, 1])
    elif proba.ndim == 2 and proba.shape[1] > 2:
        return roc_auc_score(y_true, proba, multi_class='ovr')
    else:
        return roc_auc_score(y_true, proba)


def _evaluate_tabpfn(tabpfn_model, X_arr, timeout=30):
    """Run TabPFN prediction in batches with a timeout. Returns (preds, proba, completed, duration)."""
    start = time.perf_counter()
    remaining = X_arr.copy()
    completed = 0
    preds_list = []
    proba_list = []
    while len(remaining) > 0:
        batch = remaining[:1000]
        remaining = remaining[1000:]
        preds_list.append(tabpfn_model.predict(batch))
        if hasattr(tabpfn_model, 'predict_proba'):
            proba_list.append(tabpfn_model.predict_proba(batch))
        completed += len(batch)
        if time.perf_counter() - start > timeout:
            print(f"TabPFN inference timed out after {timeout}s. Stopped at {completed} samples.")
            break
    duration = time.perf_counter() - start
    preds = np.concatenate(preds_list) if preds_list else np.array([])
    proba = np.concatenate(proba_list) if proba_list else None
    return preds, proba, completed, duration


def test(X: pd.DataFrame, y: pd.Series, model_folder: str):
    """Load saved models from training and evaluate them on the test set.
    Automatically discovers all TabPFN models with different context sizes.
    Args:
        X (pd.DataFrame): Test features
        y (pd.Series): Test targets
        model_folder (str): Folder where the trained model is saved
    """

    task_type, _ = infer_task_and_metric(y)
    y_arr = y.to_numpy().reshape(-1)

    # ── xRFM ──────────────────────────────────────────────
    print("Loading xRFM model...")
    xrfm_model = xRFM(device=torch.device('cpu'))
    state_dict = torch.load(f"{model_folder}/xrfm_model.pt", map_location="cpu", weights_only=False)
    xrfm_X_train_path = Path(model_folder) / "xrfm_X_train.npy"
    if xrfm_X_train_path.exists():
        xrfm_X_train = torch.as_tensor(np.load(xrfm_X_train_path), dtype=torch.float32)
    else:
        raise FileNotFoundError(
            f"{xrfm_X_train_path} not found. Re-run training so the exact fit "
            "matrix is saved alongside the model."
        )
    xrfm_model.load_state_dict(state_dict, X_train=xrfm_X_train)

    print("Evaluating xRFM...")
    start = time.perf_counter()
    xrfm_preds_raw = np.asarray(xrfm_model.predict(X.to_numpy(dtype=np.float32)))
    xrfm_duration = time.perf_counter() - start
    xrfm_time_per_sample = xrfm_duration / len(X)

    # If xRFM returns a 2-D probability matrix, take argmax for class labels
    if xrfm_preds_raw.ndim > 1 and xrfm_preds_raw.shape[1] > 1:
        xrfm_preds = np.argmax(xrfm_preds_raw, axis=1)
    else:
        xrfm_preds = xrfm_preds_raw.reshape(-1)

    # ── XGBoost ───────────────────────────────────────────
    print("Loading XGBoost model...")
    if task_type == "categorical":
        xgb_model = xgb.XGBClassifier()
    else:
        xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(f"{model_folder}/xgboost_model.json")

    print("Evaluating XGBoost...")
    start = time.perf_counter()
    xgb_preds = xgb_model.predict(X)
    xgb_proba = xgb_model.predict_proba(X) if hasattr(xgb_model, 'predict_proba') else None
    xgb_duration = time.perf_counter() - start
    xgb_time_per_sample = xgb_duration / len(X)

    # ── TabPFN (one per context size) ─────────────────────
    model_dir = Path(model_folder)
    tabpfn_files = sorted(model_dir.glob("tabpfn_model_ctx*.tabpfn_fit"))
    # Fallback: also look for the old single-model name
    if not tabpfn_files:
        old = model_dir / "tabpfn_model.tabpfn_fit"
        if old.exists():
            tabpfn_files = [old]

    X_arr = X.to_numpy(dtype=np.float32)

    tabpfn_results = {}  # ctx_size -> dict of metrics
    for fpath in tabpfn_files:
        m = re.search(r"ctx(\d+)", fpath.stem)
        ctx_label = int(m.group(1)) if m else "default"
        print(f"Loading & evaluating TabPFN (ctx={ctx_label})...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tabpfn_model = load_fitted_tabpfn_model(str(fpath), device=device)
        preds, proba, completed, duration = _evaluate_tabpfn(tabpfn_model, X_arr)
        tabpfn_results[ctx_label] = {
            "preds": preds, "proba": proba,
            "completed": completed, "duration": duration,
            "time_per_sample": duration / max(completed, 1),
        }

    # ── Print results ─────────────────────────────────────
    if task_type == "regression":
        xrfm_mse = np.mean((xrfm_preds - y_arr) ** 2)
        xgb_mse = np.mean((xgb_preds.reshape(-1) - y_arr) ** 2)

        print("\nxRFM performance----------------------------------")
        print(f"  MSE: {xrfm_mse:.4f}")
        print(f"  Inference Time per Sample: {xrfm_time_per_sample:.6f}s")
        print("\nXGBoost performance-------------------------------")
        print(f"  MSE: {xgb_mse:.4f}")
        print(f"  Inference Time per Sample: {xgb_time_per_sample:.6f}s")
        for ctx, r in tabpfn_results.items():
            c = r["completed"]
            mse = np.mean((r["preds"].reshape(-1) - y_arr[:c]) ** 2)
            print(f"\nTabPFN (ctx={ctx}) performance--------------------")
            print(f"  MSE: {mse:.4f}  (evaluated on {c}/{len(y_arr)} samples)")
            print(f"  Inference Time per Sample: {r['time_per_sample']:.6f}s")
    else:
        xrfm_acc = np.mean(xrfm_preds == y_arr)
        xgb_acc = np.mean(np.asarray(xgb_preds).reshape(-1) == y_arr)
        xrfm_auc = _safe_auc(y_arr, xrfm_preds_raw if xrfm_preds_raw.ndim > 1 else xrfm_preds_raw.reshape(-1))
        xgb_auc = _safe_auc(y_arr, xgb_proba)

        print("\nxRFM performance----------------------------------")
        print(f"  Accuracy: {xrfm_acc:.4f}")
        print(f"  AUC-ROC: {xrfm_auc:.4f}")
        print(f"  Inference Time per Sample: {xrfm_time_per_sample:.6f}s")
        print("\nXGBoost performance-------------------------------")
        print(f"  Accuracy: {xgb_acc:.4f}")
        print(f"  AUC-ROC: {xgb_auc:.4f}")
        print(f"  Inference Time per Sample: {xgb_time_per_sample:.6f}s")
        for ctx, r in tabpfn_results.items():
            c = r["completed"]
            acc = np.mean(np.asarray(r["preds"]).reshape(-1) == y_arr[:c])
            auc = _safe_auc(y_arr[:c], r["proba"][:c] if r["proba"] is not None else None)
            print(f"\nTabPFN (ctx={ctx}) performance--------------------")
            print(f"  Accuracy: {acc:.4f}  (evaluated on {c}/{len(y_arr)} samples)")
            print(f"  AUC-ROC: {auc:.4f}")
            print(f"  Inference Time per Sample: {r['time_per_sample']:.6f}s")
