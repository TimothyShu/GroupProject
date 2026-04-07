import pandas as pd


def infer_task_and_metric(y: pd.Series) -> tuple[str, str]:
    if (
        isinstance(y.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(y)
        or pd.api.types.is_string_dtype(y)
        or pd.api.types.is_bool_dtype(y)
    ):
        return "categorical", "accuracy"
    return "regression", "mse"