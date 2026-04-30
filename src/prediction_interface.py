import pandas as pd


def predict_single_case(model, case: dict) -> float:
    """Predict the duration of one surgical case."""
    case_df = pd.DataFrame([case])
    prediction = model.predict(case_df)[0]
    return float(prediction)


def format_duration(minutes: float) -> str:
    """Format minutes as hours and minutes."""
    hours = int(minutes // 60)
    minutes_part = int(round(minutes % 60))
    return f"{hours}h {minutes_part}m"
