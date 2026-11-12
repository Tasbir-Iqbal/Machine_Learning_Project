from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold, train_test_split

from train_models import MODELS as ORIGINAL_MODELS
from train_additional_models import MODELS as ADDITIONAL_MODELS

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "validation_importance"
CHARTS = OUTPUT / "charts"

CUTOFF = 28
N_SPLITS = 5
N_REPEATS = 3
RANDOM_STATE = 42

# Validation subsampling limits KNN's computational cost.
# Set to None to inspect every validation enrolment.
MAX_VALIDATION_ROWS = 1000

KEYS = ["code_module", "code_presentation", "id_student"]

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "k_nearest_neighbors": "K-Nearest Neighbors",
    "naive_bayes": "Gaussian Naive Bayes",
    "gradient_boosting": "Gradient Boosting",
}

MODELS = {**ORIGINAL_MODELS, **ADDITIONAL_MODELS}


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


def select_validation_rows(validation_indices, y, fold):
    if (
        MAX_VALIDATION_ROWS is None
        or len(validation_indices) <= MAX_VALIDATION_ROWS
    ):
        return validation_indices

    selected, _ = train_test_split(
        validation_indices,
        train_size=MAX_VALIDATION_ROWS,
        stratify=y.iloc[validation_indices],
        random_state=RANDOM_STATE + fold,
    )

    return np.sort(selected)


def draw_importance(summary, model_name):
    top = (
        summary.sort_values("importance_mean", ascending=False)
        .head(15)
        .sort_values("importance_mean")
    )

    fig, axis = plt.subplots(figsize=(11, 7))

    axis.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"],
        color="#3783a5",
        capsize=2,
    )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel(
        "Validation average-precision decrease after shuffling"
    )
    axis.set_title(
        f"{MODEL_LABELS[model_name]}: Validation Permutation Importance\n"
        "Error bars: variability across folds and shuffle repeats"
    )
    axis.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    path = CHARTS / f"validation_importance_{model_name}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)

    data, y, groups = load_data()

    splitter = GroupKFold(n_splits=N_SPLITS)
    splits = list(splitter.split(data, y, groups=groups))

    print(f"Enrolments: {len(data):,}", flush=True)
    print(f"Unique students: {groups.nunique():,}", flush=True)
    print(f"Models: {len(MODELS)}", flush=True)
    print(f"Validation folds: {N_SPLITS}", flush=True)
    print(f"Shuffle repeats per feature: {N_REPEATS}", flush=True)
    print(
        f"Maximum inspected rows per fold: {MAX_VALIDATION_ROWS}",
        flush=True,
    )

    all_summaries = []
    all_baselines = []
    sampling_rows = []

    for fold, (_, validation_indices) in enumerate(splits, start=1):
        selected = select_validation_rows(validation_indices, y, fold)

        for position in selected:
            row = data.iloc[position]
            sampling_rows.append({
                "fold": fold,
                "code_module": row["code_module"],
                "code_presentation": row["code_presentation"],
                "id_student": row["id_student"],
            })

    pd.DataFrame(sampling_rows).to_csv(
        OUTPUT / "inspected_validation_enrolments.csv",
        index=False,
    )

    for model_name, template in MODELS.items():
        print(
            f"\nMODEL: {MODEL_LABELS[model_name]}",
            flush=True,
        )

        columns = get_input_columns(template)
        missing = [c for c in columns if c not in data.columns]

        if missing:
            raise ValueError(f"{model_name}: missing features {missing}")

        X = data[columns].copy()
        raw_rows = []

        for fold, (train_indices, validation_indices) in enumerate(
            splits, start=1
        ):
            train_students = set(groups.iloc[train_indices])
            validation_students = set(groups.iloc[validation_indices])

            if train_students.intersection(validation_students):
                raise ValueError("Student overlap between folds.")

            inspected_indices = select_validation_rows(
                validation_indices, y, fold
            )

            X_train = X.iloc[train_indices]
            y_train = y.iloc[train_indices]
            X_validation = X.iloc[inspected_indices]
            y_validation = y.iloc[inspected_indices]

            model = clone(template)

            # One parallelism level only, to limit Windows memory use.
            if "model__n_jobs" in model.get_params():
                model.set_params(model__n_jobs=1)

            print(
                f"  Fold {fold}/{N_SPLITS}: fitting on "
                f"{len(train_indices):,}, inspecting "
                f"{len(inspected_indices):,} validation enrolments...",
                flush=True,
            )

            model.fit(X_train, y_train)

            probability = model.predict_proba(X_validation)[:, 1]
            baseline = average_precision_score(
                y_validation, probability
            )

            all_baselines.append({
                "model": model_name,
                "fold": fold,
                "training_enrolments": len(train_indices),
                "full_validation_enrolments": len(validation_indices),
                "inspected_enrolments": len(inspected_indices),
                "validation_average_precision": baseline,
            })

            result = permutation_importance(
                model,
                X_validation,
                y_validation,
                scoring="average_precision",
                n_repeats=N_REPEATS,
                random_state=RANDOM_STATE + fold,
                n_jobs=1,
            )

            for feature_index, feature in enumerate(columns):
                for repeat in range(N_REPEATS):
                    raw_rows.append({
                        "model": model_name,
                        "feature": feature,
                        "fold": fold,
                        "repeat": repeat + 1,
                        "importance": float(
                            result.importances[feature_index, repeat]
                        ),
                    })

            print(
                f"  Fold {fold} completed. "
                f"Validation AP: {baseline:.3f}",
                flush=True,
            )

        raw = pd.DataFrame(raw_rows)
        raw.to_csv(
            OUTPUT / f"raw_importance_{model_name}.csv",
            index=False,
        )

        summary = (
            raw.groupby("feature", as_index=False)
            .agg(
                importance_mean=("importance", "mean"),
                importance_std=("importance", "std"),
            )
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )

        summary["rank"] = np.arange(1, len(summary) + 1)
        summary["model"] = model_name
        summary["folds"] = N_SPLITS
        summary["repeats_per_fold"] = N_REPEATS

        summary.to_csv(
            OUTPUT / f"importance_{model_name}.csv",
            index=False,
        )

        draw_importance(summary, model_name)
        all_summaries.append(summary)

        pd.concat(all_summaries, ignore_index=True).to_csv(
            OUTPUT / "all_model_validation_importance.csv",
            index=False,
        )

        pd.DataFrame(all_baselines).to_csv(
            OUTPUT / "validation_fold_scores.csv",
            index=False,
        )

        print("\nTop 10 validation importances:")
        print(
            summary[
                ["rank", "feature", "importance_mean", "importance_std"]
            ].head(10).round(4).to_string(index=False),
            flush=True,
        )

    settings = {
        "cutoff_day": CUTOFF,
        "splitter": "GroupKFold",
        "group": "id_student",
        "folds": N_SPLITS,
        "repeats_per_fold": N_REPEATS,
        "maximum_inspected_rows_per_fold": MAX_VALIDATION_ROWS,
        "validation_sampling": "Stratified by outcome within each fold",
        "scoring": "average_precision",
        "random_state": RANDOM_STATE,
        "models": list(MODELS),
        "notes": [
            "Preprocessing is fitted only on training folds.",
            "Importance uses only held-out validation enrolments.",
            "The same inspected enrolments are used across models.",
            "Standard deviations are descriptive, not confidence intervals.",
            "Correlated and derived features can affect individual rankings.",
            "Validation subsample scores do not replace full-cohort OOF metrics.",
            "No original baseline files are overwritten.",
        ],
    }

    (OUTPUT / "settings.json").write_text(
        json.dumps(settings, indent=2),
        encoding="utf-8",
    )

    print(f"\nCompleted. Outputs saved to: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()