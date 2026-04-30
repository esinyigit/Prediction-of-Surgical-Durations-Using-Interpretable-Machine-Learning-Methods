from pathlib import Path

import pandas as pd


TARGET_COLUMN = "duration_minutes"


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a surgical duration dataset from a CSV file."""
    return pd.read_csv(path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply simple cleaning steps used by the example workflow."""
    cleaned = df.copy()

    text_columns = ["surgery_name", "medical_unit", "surgeon", "anaesthesiologist"]
    for column in text_columns:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].astype(str).str.strip()

    numeric_columns = ["patient_age", TARGET_COLUMN]
    for column in numeric_columns:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=[TARGET_COLUMN])
    return cleaned


def split_features_target(df: pd.DataFrame):
    """Split a dataframe into model features and target values."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in dataset.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y
