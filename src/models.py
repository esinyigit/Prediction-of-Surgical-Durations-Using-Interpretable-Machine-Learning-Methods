from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from .features import build_preprocessor


def build_model(random_state: int = 42) -> Pipeline:
    """Build the final XGBoost regression pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=random_state,
                    objective="reg:squarederror",
                ),
            ),
        ]
    )
