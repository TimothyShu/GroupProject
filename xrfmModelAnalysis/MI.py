import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from utils import infer_task_and_metric

def calculateMI(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Calculate mutual information between each feature in X and the target y, return a Series with the MI values indexed by feature name"""
    
    task_type, _ = infer_task_and_metric(y)
    if task_type == "classification":
        mi = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
    else:
        mi = mutual_info_regression(X, y, discrete_features='auto', random_state=42)
    mi_series = pd.Series(mi, index=X.columns)
    return mi_series