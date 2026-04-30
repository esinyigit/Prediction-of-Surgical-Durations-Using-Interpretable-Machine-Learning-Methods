from pathlib import Path

from sklearn.model_selection import train_test_split

from .data_preprocessing import clean_dataset, load_dataset, split_features_target
from .evaluation import (
    classify_duration_group,
    metrics_by_duration_group,
    prediction_table,
    regression_metrics,
)
from .models import build_model
from .prediction_interface import format_duration, predict_single_case


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "sample_synthetic_surgeries.csv"
REPORTS_PATH = PROJECT_ROOT / "reports"
FIGURES_PATH = REPORTS_PATH / "figures"


def main() -> None:
    REPORTS_PATH.mkdir(exist_ok=True)
    FIGURES_PATH.mkdir(exist_ok=True)

    df = clean_dataset(load_dataset(DATA_PATH))
    X, y = split_features_target(df)

    duration_groups = y.apply(classify_duration_group)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=duration_groups,
    )

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = regression_metrics(y_test, predictions)
    predictions_df = prediction_table(y_test.reset_index(drop=True), predictions)

    metrics_path = REPORTS_PATH / "metrics.csv"
    predictions_path = REPORTS_PATH / "example_predictions.csv"
    group_metrics_path = REPORTS_PATH / "metrics_by_duration_group.csv"

    predictions_df.to_csv(predictions_path, index=False)
    metrics_by_duration_group(predictions_df).to_csv(group_metrics_path, index=False)

    with metrics_path.open("w", encoding="utf-8") as file:
        file.write("metric,value\n")
        for metric, value in metrics.items():
            file.write(f"{metric},{value:.3f}\n")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.scatter(y_test, predictions)
        plt.xlabel("Actual duration (minutes)")
        plt.ylabel("Predicted duration (minutes)")
        plt.title("Actual vs Predicted Surgical Duration")
        plt.tight_layout()
        plt.savefig(FIGURES_PATH / "actual_vs_predicted.png", dpi=150)
        plt.close()
    except ModuleNotFoundError:
        print("matplotlib is not installed, so the example figure was skipped.")

    example_case = {
        "surgery_name": "Appendectomy",
        "medical_unit": "General Surgery",
        "surgeon": "Doctor_A",
        "anaesthesiologist": "Anaesth_A",
        "patient_age": 45,
    }
    example_prediction = predict_single_case(model, example_case)

    print("Model evaluation metrics")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

    print()
    print("Example prediction")
    print(f"Predicted duration: {format_duration(example_prediction)}")
    print()
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved duration group metrics to: {group_metrics_path}")


if __name__ == "__main__":
    main()
