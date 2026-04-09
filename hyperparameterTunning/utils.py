import pandas as pd

def process_categorical_target(y: pd.Series) -> pd.Series:
    """Process the target variable if it is categorical. This will convert it to a categorical type and then to codes.

    Args:
        y (pd.Series): The target variable
    Returns:
        pd.Series: The processed target variable
    """
    task_type, _ = infer_task_and_metric(y)
    y_processed = y.copy()
    if task_type == "categorical":
        y_processed = pd.Series(pd.Categorical(y).codes, index=y.index)
    return y_processed

def infer_task_and_metric(y: pd.Series) -> tuple[str, str]:
    if (
        isinstance(y.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(y)
        or pd.api.types.is_string_dtype(y)
        or pd.api.types.is_bool_dtype(y)
    ):
        return "categorical", "accuracy"
    return "regression", "mse"