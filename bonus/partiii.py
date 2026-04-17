import os

import numpy as np
import openml
import pandas as pd
from sklearn.model_selection import train_test_split
import torch


if __name__ == "__main__":
    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))

    from normalizeFeatures import normalizeFeatures
    from utils import process_categorical_target, infer_task_and_metric
    from bonus.train import train
    from bonus.compareSplitDirection import compare_split_direction
    from bonus.compareAccuracy import compare_model_accuracy

    np.random.seed(42)
    torch.manual_seed(42)


    openml.config.apikey =  os.getenv("OPENML_KEY")
    # Load a binary classification dataset
    dataset = openml.datasets.get_dataset('colleges')
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format='dataframe'
    )
    
    # Drop rows where target is NaN (about 40% of dataset)
    mask = y.notna()
    X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

    y = process_categorical_target(y)

    target_type, metric = infer_task_and_metric(y)

    print(f"Target type: {target_type}, Tuning metric: {metric}")

    # ── Feature engineering ──────────────────────────────────────────

    # 1. Drop probably useless identifiers
    drop_identifiers = ["school_name", "school_webpage", "zip", "city", "UNITID", "state"]

    # 2. Drop 100% missing columns
    drop_all_missing = ["average_age_of_entry", "percent_married", "percent_veteran"]

    all_drops = (
        drop_identifiers + drop_all_missing
    )
    X = X.drop(columns=[c for c in all_drops if c in X.columns])

    # 6. Convert string columns that are actually numeric
    for col in ["percent_female", "agege24", "faminc"]:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # 7. too many to keep, but a lot missing so probably just binary for now
    if "religious_affiliation" in X.columns:
        X["has_religious_affiliation"] = X["religious_affiliation"].notna().astype(bool)
        X = X.drop(columns=["religious_affiliation"])

    # 8. Trying to drop columns to not make xrfm crash
    missing_thresh = 0.7
    high_missing = [c for c in X.columns if X[c].isna().sum() / len(X) > missing_thresh]
    if high_missing:
        print(f"Dropping {len(high_missing)} columns with >{missing_thresh*100:.0f}% missing: {high_missing}")
        X = X.drop(columns=high_missing)

    print(f"Features after cleaning: {X.shape[1]}")

    # Ensure all columns are numeric (convert any remaining non-numeric columns to NaN)
    X = X.apply(pd.to_numeric, errors="coerce")
    # Drop any columns that are still of type object (e.g., all-NaN or mixed type columns)
    X = X.select_dtypes(include=[np.number])

    # normalize the features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = normalizeFeatures(X_train, X_test)

    # now remove 

    print(X_train.head())

    xrfm, resxrfm = train(X_train, y_train, trials=20)


    compare_split_direction(xrfm, resxrfm)

    # Create and print a map from feature index to name
    feature_index_to_name = {i: name for i, name in enumerate(X_train.columns)}
    print("\nFeature index to name map:")
    for idx, name in feature_index_to_name.items():
        print(f"  {idx}: {name}")

    compare_model_accuracy(X_test, y_test, xrfm, resxrfm, metric="mse")
