from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from train_models import MODELS as ORIGINAL_MODELS
from train_additional_models import MODELS as ADDITIONAL_MODELS

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MODELS_DIR = REPORTS / "saved_models"

CUTOFF = 28
N_SPLITS = 5
RANDOM_STATE = 42

KEYS = ["code_module", "code_presentation", "id_student"]

MODELS = {**ORIGINAL_MODELS, **ADDITIONAL_MODELS}

TARGET_MODELS = [
    "gradient_boosting",
    "random_forest",
    "logistic_regression",
    "k_nearest_neighbors",
    "naive_bayes",
]


def load_data():
    path = REPORTS / f"features_day_{CUTOFF}.csv"
    data = pd.read_csv(path)

    required = KEYS + ["final_result"]
    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if data[required].isna().any().any():
        raise ValueError("Missing identifiers or outcome labels.")

    if data.duplicated(KEYS).any():
        raise ValueError("Duplicate enrolments found.")

    allowed = {"Pass", "Distinction", "Fail", "Withdrawn"}
    if not data["final_result"].isin(allowed).all():
        raise ValueError("Unexpected final_result values.")

    y = data["final_result"].isin(["Fail", "Withdrawn"]).astype(int)
    groups = data["id_student"]

    return data, y, groups


def get_input_columns(model):
    preprocess = model.named_steps["preprocess"]

    if preprocess.remainder != "drop":
        raise ValueError(
            "This script expects explicit columns and remainder='drop'."
        )

    columns = []

    for _, transformer, selected_columns in preprocess.transformers:
        if isinstance(transformer, str) and transformer == "drop":
            continue

        if not isinstance(selected_columns, (list, tuple)):
            raise ValueError("Expected an explicit feature-column list.")

        columns.extend(selected_columns)

    return list(dict.fromkeys(columns))


def save_pipeline(model, model_name, fold, feature_columns):
    directory = MODELS_DIR / model_name
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / f"fold_{fold}_pipeline.joblib"
    joblib.dump(model, path)

    metadata = {
        "model": model_name,
        "fold": fold,
        "pipeline_path": str(path.relative_to(REPORTS)),
        "feature_columns": feature_columns,
        "cutoff_day": CUTOFF,
        "target": "at_risk",
        "target_definition": "Fail or Withdrawn",
        "prediction_type": "probability",
    }

    (directory / f"fold_{fold}_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    data, y, groups = load_data()

    splitter = GroupKFold(n_splits=N_SPLITS)
    splits = list(splitter.split(data, y, groups=groups))

    print(f"Enrolments: {len(data):,}", flush=True)
    print(f"Unique students: {groups.nunique():,}", flush=True)
    print(f"Saving models: {TARGET_MODELS}", flush=True)

    for model_name in TARGET_MODELS:
        if model_name not in MODELS:
            raise ValueError(f"Unknown model: {model_name}")

        template = MODELS[model_name]
        columns = get_input_columns(template)

        missing = [c for c in columns if c not in data.columns]

        if missing:
            raise ValueError(f"{model_name}: missing features {missing}")

        X = data[columns].copy()

        print(f"\nSaving {model_name} pipelines...", flush=True)

        for fold, (train_indices, _) in enumerate(splits, start=1):
            train_students = set(groups.iloc[train_indices])

            X_train = X.iloc[train_indices]
            y_train = y.iloc[train_indices]

            model = clone(template)

            if (
                model_name != "logistic_regression"
                and "model__n_jobs" in model.get_params()
            ):
                model.set_params(model__n_jobs=1)

            print(
                f"  Fold {fold}/{N_SPLITS}: fitting on "
                f"{len(train_indices):,} enrolments...",
                flush=True,
            )

            model.fit(X_train, y_train)
            save_pipeline(model, model_name, fold, columns)

        print(f"  Saved {N_SPLITS} pipelines for {model_name}.", flush=True)

    settings = {
        "cutoff_day": CUTOFF,
        "saved_models": TARGET_MODELS,
        "folds": N_SPLITS,
        "group": "id_student",
        "random_state": RANDOM_STATE,
        "model_format": "joblib",
        "notes": [
            "Each pipeline includes preprocessing and the fitted classifier.",
            "Pipelines are fitted only on that fold's training data.",
            "Use the matching fold's validation predictions for evaluation.",
            "Do not use a pipeline to evaluate students that appeared in its training fold.",
        ],
    }

    (MODELS_DIR / "saved_models_settings.json").write_text(
        json.dumps(settings, indent=2),
        encoding="utf-8",
    )

    print(f"\nCompleted. Models saved to: {MODELS_DIR}", flush=True)


if __name__ == "__main__":
    main()