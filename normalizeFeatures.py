import pandas as pd

from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def normalizeFeatures(X_train: pd.DataFrame, X_test: pd.DataFrame, excludeColumns = []) -> pd.DataFrame:
    """Simple function to normalize features. It will one-hot encode categorical features and standardize numerical features.
    For more complicated normalization do it before running this method and then cat the results together

    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        excludeColumns (list, optional): List of columns to exclude from normalization. Defaults to [].
    
    Returns:
        X_train (pd.DataFrame): Normalized training features
        X_test (pd.DataFrame): Normalized testing features
    """

    X_train = X_train.drop(columns=excludeColumns)
    X_test = X_test.drop(columns=excludeColumns)

    # These should be pipelines but we are only doing tests and not deploying so is fine
    transformer = make_column_transformer(
        (make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore', sparse_output=False)), 
        make_column_selector(dtype_include=['object', 'category'], dtype_exclude=['bool'])),
        (make_pipeline(SimpleImputer(strategy='median'), StandardScaler()),
        make_column_selector(dtype_include=['number'], dtype_exclude=['bool'])),
        
        remainder='passthrough'
    )

    # 2. Keep the output as a clean Pandas DataFrame
    transformer.set_output(transform="pandas")

    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)

    return X_train, X_test

if __name__ == "__main__":
    # run some quick tests
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    # Load the Titanic dataset
    titanic = fetch_openml('titanic', version=1, as_frame=True)
    X = titanic.data
    y = titanic.target
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Normalize the features
    X_train, X_test = normalizeFeatures(X_train, X_test, excludeColumns=['name', 'ticket', 'cabin'])
    print(X_train.head())
    print(X_test.head())