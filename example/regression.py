


if __name__ == "__main__":
    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))

    from normalizeFeatures import normalizeFeatures
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from hyperparameterTunning.utils import process_categorical_target
    from training import train
    from testing import test

    # Load the California housing dataset
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target

    y = process_categorical_target(y) # should do nothing as this is a regression task but just in case

    # normalize the features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = normalizeFeatures(X_train, X_test)

    # only train if the model is not already trained, this is to save time when testing the hyperparameter tuning
    if not (PROJECT_ROOT / "models" / "example" / "xrfm_model.pt").exists():
        # train the model and tune hyperparameters
        train(X_train, y_train)
    
    test(X_test, y_test, X_train, "models/example")
