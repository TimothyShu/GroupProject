import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
import torch

def _get_mse(model, X_input, y_true):
    # Convert X to tensor (ensuring a writable copy for PyTorch)
    X_arr = X_input.to_numpy() if isinstance(X_input, pd.DataFrame) else X_input
    X_tensor = torch.as_tensor(X_arr.copy(), dtype=torch.float32).to(model.device)
    
    y_arr = y_true.to_numpy() if isinstance(y_true, (pd.Series, pd.DataFrame)) else y_true
    y_tensor = torch.as_tensor(y_arr.copy(), dtype=torch.float32).to(model.device)
    
    with torch.no_grad():
        preds = model.predict(X_tensor) # Or model(X_tensor) depending on your API
        #print(f"Types: preds={type(preds)}, y_tensor={type(y_tensor)}")

        # no idea but preds is numpy array
        preds = torch.as_tensor(preds, dtype=torch.float32).to(model.device)
        mse = torch.mean((preds - y_tensor)**2).item()
    return mse

def calculatePermutationImportance(model, X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Calculate permutation importance for each feature in X with respect to the target y using the given metric, return a Series with the importance values indexed by feature name"""

    # manually calculate because sklearn nor working

    n_repeats = 10

    # Baseline performance
    baseline_mse = _get_mse(model, X, y)
    feature_names = X.columns if isinstance(X, pd.DataFrame) else range(X.shape[1])
    importances = []
    for col_idx, feature in enumerate(feature_names):
        scores = []
        for r in range(n_repeats):
            # Create a copy and shuffle only the target feature
            X_permuted = X.copy()
            if isinstance(X_permuted, pd.DataFrame):
                X_permuted[feature] = np.random.permutation(X_permuted[feature].values)
            else:
                X_permuted[:, col_idx] = np.random.permutation(X_permuted[:, col_idx])
            
            # Calculate new MSE and the "drop" in performance
            permuted_mse = _get_mse(model, X_permuted, y)
            # Importance is the increase in error
            scores.append(permuted_mse - baseline_mse)
        
        importances.append(scores)
        print(f"Done: {feature}")

    # 3. View Results
    importances_mean = np.mean(importances, axis=1)

    # normalize for for range -1 to 1
    importances_mean = importances_mean / np.max(np.abs(importances_mean))
    return pd.Series(importances_mean, index=feature_names)