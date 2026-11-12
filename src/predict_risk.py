from pathlib import Path
import argparse
import json
import numpy as np
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SAVED_MODELS = REPORTS / "saved_models"

MODEL_NAME = "gradient_boosting"
N_FOLDS = 5

RISK_TIERS = [
    (0.00, 0.30, "Low"),
    (0.30, 0.50, "Moderate"),
    (0.50, 0.70, "High"),
    (0.70, 1.01, "Very High"),
]


def load_pipelines():
    directory = SAVED_MODELS / MODEL_NAME
    pipelines = []

    for fold in range(1, N_FOLDS + 1):
        path = directory / f"fold_{fold}_pipeline.joblib"

        if not path.exists():
            raise FileNotFoundError(f"Missing pipeline: {path}")

        pipelines.append(joblib.load(path))

    return pipelines


def assign_risk(probability):
    for lower, upper, label in RISK_TIERS:
        if lower <= probability < upper:
            return label

    return "Very High"


def main():
    parser = argparse.ArgumentParser(
        description="Predict at-risk probability using saved Gradient Boosting pipelines."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to a CSV file containing the required feature columns.",
    )

    parser.add_argument(
        "--output",
        default="risk_predictions.csv",
        help="Path for the output CSV file.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data = pd.read_csv(input_path)

    metadata_path = (
        SAVED_MODELS / MODEL_NAME / "fold_1_metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required_columns = metadata["feature_columns"]

    missing = [c for c in required_columns if c not in data.columns]

    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    X = data[required_columns].copy()
    pipelines = load_pipelines()

    probabilities = []

    for index, pipeline in enumerate(pipelines, start=1):
        probability = pipeline.predict_proba(X)[:, 1]
        probabilities.append(probability)

        print(f"Pipeline {index}/{N_FOLDS} prediction completed.", flush=True)

    probability = np.mean(probabilities, axis=0)
    prediction = (probability >= 0.5).astype(int)
    risk_tier = [assign_risk(value) for value in probability]

    output = data.copy()
    output["at_risk_probability"] = probability
    output["at_risk_prediction"] = prediction
    output["risk_tier"] = risk_tier

    output.to_csv(output_path, index=False)

    print(f"\nPredictions saved to: {output_path}", flush=True)
    print("\nRisk-tier counts:")
    print(output["risk_tier"].value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()