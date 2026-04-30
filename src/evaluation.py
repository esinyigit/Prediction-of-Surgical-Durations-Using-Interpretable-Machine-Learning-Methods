import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error


def classify_duration_group(minutes: float) -> str:
    """Classify surgeries into short, medium, and long duration groups."""
    if minutes < 60:
        return "short"
    if minutes <= 120:
        return "medium"
    return "long"


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Calculate common regression metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "median_absolute_error": float(median_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def prediction_table(y_true, y_pred) -> pd.DataFrame:
    """Create a table comparing true and predicted durations."""
    return pd.DataFrame(
        {
            "actual_duration_minutes": y_true,
            "predicted_duration_minutes": y_pred,
            "duration_group": [classify_duration_group(value) for value in y_true],
            "absolute_error_minutes": np.abs(np.asarray(y_true) - np.asarray(y_pred)),
        }
    )


def metrics_by_duration_group(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize prediction error for short, medium, and long surgeries."""
    return (
        predictions.groupby("duration_group", as_index=False)
        .agg(
            mean_absolute_error_minutes=("absolute_error_minutes", "mean"),
            median_absolute_error_minutes=("absolute_error_minutes", "median"),
            number_of_cases=("absolute_error_minutes", "size"),
        )
        .sort_values("duration_group")
    )
