if __name__ == "__main__":
    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))

    from normalizeFeatures import normalizeFeatures
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from utils import process_categorical_target
    from models.training import train
    from models.testing import test

    # Load the California housing dataset
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target

    y = process_categorical_target(y) # should do nothing as this is a regression task but just in case

    # normalize the features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = normalizeFeatures(X_train, X_test)

    train(X_train, y_train, "models/example")
    
    test(X_test, y_test, X_train, "models/example")
