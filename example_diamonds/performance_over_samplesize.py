
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
import openml


if __name__ == "__main__":
    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))

    from normalizeFeatures import normalizeFeatures
    from utils import process_categorical_target, infer_task_and_metric
    from models.training import train
    from models.testing import test

    load_dotenv() 
    
    openml.config.apikey =  os.getenv("OPENML_KEY")
    # Load a binary classification dataset
    dataset = openml.datasets.get_dataset('diamonds')
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format='dataframe'
    )
    
    y = process_categorical_target(y)

    target_type, metric = infer_task_and_metric(y)

    print(f"Target type: {target_type}, Tuning metric: {metric}")

    # Shuffle the data before splitting into subsets
    import numpy as np
    shuffled_indices = np.random.permutation(X.index)
    X = X.loc[shuffled_indices].reset_index(drop=True)
    y = y.loc[shuffled_indices].reset_index(drop=True)

    total_samples = X.shape[0]
    num_trials = 10
    subset_sizes = np.linspace(int(total_samples/num_trials), total_samples, num=num_trials, dtype=int)

    for i, subset_size in enumerate(subset_sizes, 1):
        # Use only the first subset_size samples
        X_sub = X.iloc[:subset_size]
        y_sub = y.iloc[:subset_size]
        X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=0.2, random_state=42)
        X_train, X_test = normalizeFeatures(X_train, X_test)

        train(X_train, y_train, "models/temp_diamonds", refit=True, hyperparameter_tuning_folds=3, trials=50)
        test(X_test, y_test, "models/temp_diamonds")

        print(f"\n=== Trial {i}/{num_trials}: Using {subset_size} samples ===")

        # Prompt user to continue
        input(f"Completed trial {i}. Press Enter to continue to the next trial...")
