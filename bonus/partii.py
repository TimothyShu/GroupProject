import os
from dotenv import load_dotenv
import numpy as np
import openml
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

    load_dotenv() 
    
    openml.config.apikey =  os.getenv("OPENML_KEY")
    # Load a binary classification dataset
    dataset = openml.datasets.get_dataset('credit-g')
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format='dataframe'
    )
    
    y = process_categorical_target(y)

    target_type, metric = infer_task_and_metric(y)

    # normalize the features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = normalizeFeatures(X_train, X_test)

    xrfm, resxrfm = train(X_train, y_train, trials=30)

    compare_split_direction(xrfm, resxrfm)

    compare_model_accuracy(X_test, y_test, xrfm, resxrfm, metric=metric)