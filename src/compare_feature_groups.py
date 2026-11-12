from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from train_models import MODELS as ORIGINAL_MODELS
from train_additional_models import MODELS as ADDITIONAL_MODELS

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "feature_group_comparison"
CHARTS = OUTPUT / "charts"

CUTOFF = 28
N_SPLITS = 5
RANDOM_STATE = 42
THRESHOLD = 0.5

KEYS = ["code_module", "code_presentation", "id_student"]

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "k_nearest_neighbors": "K-Nearest Neighbors",
    "naive_bayes": "Gaussian Naive Bayes",
    "gradient_boosting": "Gradient Boosting",
}

MODELS = {**ORIGINAL_MODELS, **ADDITIONAL_MODELS}

ACTIVITY_FEATURES = [
    "activity_rate",
    "active_days",
    "no_activity",
    "days_since_last_activity",
    "clicks_days_21_28",
]

ASSESSMENT_FEATURES = [
    "early_mean_score",
    "early_score_count",
    "early_score_missing",
    "missed_assessments",
    "assessment_completion_rate",
    "late_submissions",
    "banked_assessments",
]

CONTEXT_FEATURES = [
    "code_module",
    "code_presentation",
    "gender",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
    "num_of_prev_attempts",
    "studied_credits",
]

SCORE_FEATURES = [
    "early_mean_score",
    "early_score_count",
    "early_score_missing",
]

CONFIGURATIONS = {
    "activity_context": ACTIVITY_FEATURES + CONTEXT_FEATURES,
    "assessment_context": ASSESSMENT_FEATURES + CONTEXT_FEATURES,
    "full": ACTIVITY_FEATURES + ASSESSMENT_FEATURES + CONTEXT_FEATURES,
    "full_no_scores": [
        feature
        for feature in ACTIVITY_FEATURES + ASSESSMENT_FEATURES + CONTEXT_FEATURES
        if feature not in SCORE_FEATURES
    ],
}


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


def make_model(model_name, template, columns):
    model = clone(template)
    preprocess = model.named_steps["preprocess"]

    if preprocess.remainder != "drop":
        raise ValueError(
            "This script expects explicit columns and remainder='drop'."
        )

    requested = set(columns)
    transformers = []

    for name, transformer, selected_columns in preprocess.transformers:
        if isinstance(transformer, str) and transformer == "drop":
            continue

        if not isinstance(selected_columns, (list, tuple)):
            raise ValueError(
                "Expected an explicit feature-column list."
            )

        retained = [
            feature
            for feature in selected_columns
            if feature in requested
        ]

        if retained:
            transformers.append((name, transformer, retained))

    model.named_steps["preprocess"].transformers = transformers
    model.named_steps["preprocess"].remainder = "drop"

    if (
        model_name != "logistic_regression"
        and "model__n_jobs" in model.get_params()
    ):
        model.set_params(model__n_jobs=1)

    return model


def draw_group_chart(summary):
    labels = {
        "activity_context": "Activity + Context",
        "assessment_context": "Assessments + Context",
        "full": "Full Model",
        "full_no_scores": "Full Without Early Scores",
    }

    summary = summary.copy()
    summary["configuration_label"] = summary["configuration"].map(labels)

    fig, axis = plt.subplots(figsize=(11, 6))

    for model_name in MODEL_LABELS:
        subset = summary[summary["model"] == model_name]
        subset = subset.set_index("configuration_label").reindex(
            list(labels.values())
        )

        axis.plot(
            subset.index,
            subset["roc_auc_mean"],
            marker="o",
            linewidth=2,
            label=MODEL_LABELS[model_name],
        )

    axis.set_ylabel("Mean out-of-fold ROC AUC")
    axis.set_title(
        "Feature-Group Comparison Across Five Student-Grouped Folds"
    )
    axis.set_ylim(0.5, 0.85)
    axis.grid(alpha=0.2)
    axis.legend(
        title="Classifier",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    fig.tight_layout()
    fig.savefig(
        CHARTS / "feature_group_roc_auc.png",
        dpi=220,
        bbox_inches="tight",
    )
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
    print(f"Feature configurations: {len(CONFIGURATIONS)}", flush=True)
    print(f"Student-grouped folds: {N_SPLITS}", flush=True)

    rows = []

    for model_name, template in MODELS.items():
        available = get_input_columns(template)

        for configuration, columns in CONFIGURATIONS.items():
            missing = [c for c in columns if c not in available]

            if missing:
                raise ValueError(
                    f"{model_name} / {configuration}: missing {missing}"
                )

            X = data[columns].copy()
            fold_rows = []

            print(
                f"\n{MODEL_LABELS[model_name]} — {configuration}",
                flush=True,
            )

            for fold, (train_indices, validation_indices) in enumerate(
                splits, start=1
            ):
                train_students = set(groups.iloc[train_indices])
                validation_students = set(groups.iloc[validation_indices])

                if train_students.intersection(validation_students):
                    raise ValueError("Student overlap between folds.")

                X_train = X.iloc[train_indices]
                y_train = y.iloc[train_indices]
                X_validation = X.iloc[validation_indices]
                y_validation = y.iloc[validation_indices]

                model = make_model(model_name, template, columns)
                model.fit(X_train, y_train)

                probability = model.predict_proba(X_validation)[:, 1]
                prediction = (probability >= THRESHOLD).astype(int)

                fold_rows.append({
                    "model": model_name,
                    "configuration": configuration,
                    "fold": fold,
                    "roc_auc": roc_auc_score(y_validation, probability),
                    "average_precision": average_precision_score(
                        y_validation, probability
                    ),
                    "brier_score": brier_score_loss(
                        y_validation, probability
                    ),
                    "precision": precision_score(
                        y_validation, prediction, zero_division=0
                    ),
                    "recall": recall_score(
                        y_validation, prediction, zero_division=0
                    ),
                    "f1": f1_score(
                        y_validation, prediction, zero_division=0
                    ),
                })

                print(
                    f"  Fold {fold}: ROC AUC "
                    f"{fold_rows[-1]['roc_auc']:.3f}, "
                    f"AP {fold_rows[-1]['average_precision']:.3f}",
                    flush=True,
                )

            rows.extend(fold_rows)

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT / "fold_results.csv", index=False)

    summary = (
        results.groupby(["model", "configuration"], as_index=False)
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            average_precision_mean=("average_precision", "mean"),
            average_precision_std=("average_precision", "std"),
            brier_score_mean=("brier_score", "mean"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
        )
    )

    summary.to_csv(OUTPUT / "summary.csv", index=False)

    draw_group_chart(summary)

    settings = {
        "cutoff_day": CUTOFF,
        "splitter": "GroupKFold",
        "group": "id_student",
        "folds": N_SPLITS,
        "threshold": THRESHOLD,
        "random_state": RANDOM_STATE,
        "configurations": CONFIGURATIONS,
        "notes": [
            "Preprocessing is fitted only on training folds.",
            "The same student-grouped folds are used for every configuration.",
            "Context features are held constant across the first three configurations.",
            "The score-free variant removes early assessment scores but retains other assessment features.",
            "No original baseline outputs are overwritten.",
        ],
    }

    (OUTPUT / "settings.json").write_text(
        json.dumps(settings, indent=2),
        encoding="utf-8",
    )

    print("\nMean out-of-fold results:")
    print(
        summary[
            [
                "model",
                "configuration",
                "roc_auc_mean",
                "average_precision_mean",
                "brier_score_mean",
                "recall_mean",
                "f1_mean",
            ]
        ]
        .round(3)
        .to_string(index=False),
        flush=True,
    )

    print(f"\nCompleted. Outputs saved to: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()