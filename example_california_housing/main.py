from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing


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

    # Load a binary classification dataset
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target
    
    y = process_categorical_target(y)

    target_type, metric = infer_task_and_metric(y)

    print(f"Target type: {target_type}, Tuning metric: {metric}")

    # normalize the features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = normalizeFeatures(X_train, X_test)

    train(X_train, y_train, "models/california_housing", refit=False, hyperparameter_tuning_folds=3, trials=50)
    
    test(X_test, y_test, "models/california_housing")
